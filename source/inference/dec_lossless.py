import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
import os
import random

import inference.arithmetic_coding as ac
import Model.Model_lossless as Model
import copy
from inference.utils import (find_min_and_max, img2patch, img2patch_padding,
                   model_lambdas, patch2img, qp_shifts, rgb2yuv_lossless,
                   yuv2rgb_lossless)


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def dec_binary(dec, bin_num):
    value = 0
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        dec_c = dec.read(freqs)
        value = value + (2**(bin_num-1-i))*dec_c
    return value

def dec_band(subband,temp_context,decomp_level,use_context,freqs_resolution,entropy_model,dec,code_block_size,subband_w,subband_h,padding_sub_w,padding_sub_h,yuv_low_bound,yuv_high_bound,shift_min,shift_max,min_v):
    coded_coe_num = 0
    tmp_stride = subband_w + padding_sub_w
    tmp_hor_num = tmp_stride // code_block_size
    paddings = (0, padding_sub_w, 0, padding_sub_h)
    dec_band = F.pad(subband, paddings, "constant")
    dec_band = img2patch(dec_band, code_block_size, code_block_size, code_block_size)
    
    if use_context:
        if padding_sub_w >= temp_context.shape[3] or padding_sub_h >= temp_context.shape[2]:
            context = F.pad(temp_context, paddings, "constant")
        else:
            context = F.pad(temp_context, paddings, "reflect")
    
    paddings = (6, 6, 6, 6)
    dec_band = F.pad(dec_band, paddings, "constant")
    if use_context:
        context = F.pad(context, paddings, "reflect")
        context = img2patch_padding(context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
        
    for h_i in range(code_block_size):
        for w_i in range(code_block_size):
            cur_ct = dec_band[:, :, h_i:h_i + 13, w_i:w_i + 13]
            
            if use_context:
                cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                prob = entropy_model(cur_ct, cur_context,yuv_low_bound, yuv_high_bound,decomp_level)
            else:
                prob = entropy_model(cur_ct, yuv_low_bound, yuv_high_bound)
                
            prob = prob.cpu().data.numpy()

            index = []

            prob = prob * freqs_resolution
            prob = prob.astype(np.int64)

            for sample_idx, prob_sample in enumerate(prob):
                coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                            h_i * tmp_stride + \
                            ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                            w_i
                if (coe_id % tmp_stride) < subband_w and (coe_id // tmp_stride) < subband_h:
                    yuv_flag = sample_idx % 3
                    
                    if shift_min[yuv_flag] < shift_max[yuv_flag]:
                        freqs = ac.SimpleFrequencyTable(prob_sample[shift_min[yuv_flag]:shift_max[yuv_flag] + 1])
                        dec_c = dec.read(freqs) + min_v[yuv_flag]
                    else:
                        dec_c = min_v[yuv_flag]
                    coded_coe_num = coded_coe_num + 1
                    index.append(dec_c)
                else:
                    index.append(0)
            dec_band[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
    band = dec_band[:, :, 6:-6, 6:-6]
    band = patch2img(band, subband_h + padding_sub_h, subband_w + padding_sub_w)
    band = band[:, :, 0:subband_h, 0:subband_w]
    return band,coded_coe_num

def dec_lossless(args, bin_name, dec, freqs_resolution, logfile):
    
    seed=0
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
        
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    decomp_levels = args.decomp_levels
    code_block_size = args.code_block_size

    with torch.no_grad():

        model_qp = 27

        qp_shift = 0
        init_scale = qp_shifts[model_qp][qp_shift]
        print(init_scale)
        logfile.write(str(init_scale) + '\n')
        logfile.flush()

        # reload main model
        checkpoint = torch.load(args.model_path + '/' + str(model_lambdas[model_qp]) + '_lossless.pth')

        all_part_dict = checkpoint['state_dict']

        models_dict = {}
        is_train=False
        models_dict['transform'] = Model.Model(args.decomp_levels,args.alpha,False,args.lifting_struct,args.non_linear_arch,args.wavelet,is_train,False,args.nec)
        models_dict['entropy_LL'] = Model.CodingLL_lossless(args.nec)
        models_dict['entropy_HL'] = Model.CodingHL_lossless(args.decomp_levels,args.nec)
        models_dict['entropy_LH'] = Model.CodingLH_lossless(args.decomp_levels,args.nec)
        models_dict['entropy_HH'] = Model.CodingHH_lossless(args.decomp_levels,args.nec)


        models_dict_update = {}
        for key, model in models_dict.items():
            myparams_dict = model.state_dict()
            if key == "context":
                part_dict = {k.split("context.")[-1]: v for k, v in all_part_dict.items() if k.split("context.")[-1] in myparams_dict}
            else:
                part_dict = {k: v for k, v in all_part_dict.items() if k in myparams_dict}
            # part_dict = {k: v for k, v in all_part_dict.items() if k in myparams_dict}
            myparams_dict.update(part_dict)
            model.load_state_dict(myparams_dict)
            if torch.cuda.is_available():
                model = model.cuda()
                model.eval()
            models_dict_update[key] = model
        models_dict.update(models_dict_update)
        
        for i in range(len(models_dict['transform'].wavelet_transform)):
            print(str(i+1)+" Level of decomposition")
            for name, layer in models_dict['transform'].wavelet_transform[i].lifting.named_modules(): 
                if hasattr(layer, 'alpha'):
                    print(name+"_alpha: "+str(round(layer.alpha.item(),4)))

        print('Load pre-trained model succeed!')
        logfile.write('Load pre-trained model succeed!' + '\n')
        logfile.flush()

        height = dec_binary(dec, 15)
        width = dec_binary(dec, 15)

        pad_h = int(np.ceil(height / (2**(decomp_levels)))) * (2**(decomp_levels)) - height
        pad_w = int(np.ceil(width / (2**(decomp_levels)))) * (2**(decomp_levels)) - width

        LL = torch.zeros(3, 1, (height + pad_h) // 2**(decomp_levels), (width + pad_w) // 2**(decomp_levels)).cuda()
        HL_list = []
        LH_list = []
        HH_list = []
        down_scales = [2, 4, 8, 16]
        for i in range(decomp_levels):
           
            HL_list.append(torch.zeros(3, 1, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]).cuda())
            LH_list.append(torch.zeros(3, 1, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]).cuda())
            HH_list.append(torch.zeros(3, 1, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]).cuda())

        min_v = np.zeros(shape=(3, decomp_levels*3+1), dtype=np.int)
        max_v = np.zeros(shape=(3, decomp_levels*3+1), dtype=np.int)
        for i in range(3):
            for j in range(decomp_levels*3+1):
                min_v[i, j] = dec_binary(dec, 15) - 6016
                max_v[i, j] = dec_binary(dec, 15) - 6016
        yuv_low_bound = min_v.min(axis=0)
        yuv_high_bound = max_v.max(axis=0)
        shift_min = min_v - yuv_low_bound
        shift_max = max_v - yuv_low_bound

        subband_h = [ (height + pad_h) // 2**(i+1) for i in range(decomp_levels) ]
        subband_w = [ (width + pad_w) // 2**(i+1) for i in range(decomp_levels) ] 
        padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_h]
        padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_w]

        coded_coe_num = 0
        # decompress LL
        
        LL,coded_num = dec_band(LL,torch.zeros(1),decomp_levels-1,False,freqs_resolution,models_dict['entropy_LL'],
                 dec,code_block_size,subband_w[decomp_levels-1],subband_h[decomp_levels-1],
                 padding_sub_w[decomp_levels-1],padding_sub_h[decomp_levels-1],yuv_low_bound[0],
                 yuv_high_bound[0],shift_min[:, 0],shift_max[:, 0],min_v[:, 0])
        
        coded_coe_num+=coded_num
        
        print('LL decoded')
        
        LL_sn = copy.deepcopy(LL)

        for i in range(decomp_levels):
            j = decomp_levels - 1 - i
            

            temp_context_HL = LL
            
            band_temp,coded_num = dec_band(HL_list[j],temp_context_HL,j,True,freqs_resolution,models_dict['entropy_HL'],
                 dec,code_block_size,subband_w[j],subband_h[j],
                 padding_sub_w[j],padding_sub_h[j],yuv_low_bound[3 * j + 1],
                 yuv_high_bound[3 * j + 1],shift_min[:, 3 * j + 1],shift_max[:, 3 * j + 1],min_v[:, 3 * j + 1])
            
            HL_list[j]=band_temp
            coded_coe_num+=coded_num
    
            print('HL' + str(j) + ' decoded')
            
            # decompress LH

            temp_context_LH = torch.cat((LL, HL_list[j]), dim=1)
            
            band_temp,coded_num = dec_band(LH_list[j],temp_context_LH,j,True,freqs_resolution,models_dict['entropy_LH'],
                 dec,code_block_size,subband_w[j],subband_h[j],
                 padding_sub_w[j],padding_sub_h[j],yuv_low_bound[3 * j + 2],
                 yuv_high_bound[3 * j + 2],shift_min[:, 3 * j + 2],shift_max[:, 3 * j + 2],min_v[:, 3 * j + 2])
            
            LH_list[j]=band_temp
            coded_coe_num+=coded_num
            
            print('LH' + str(j) + ' decoded')
            
            # decompress HH
            temp_context_HH = torch.cat((LL, HL_list[j], LH_list[j]), dim=1)  
            
            band_temp,coded_num = dec_band(HH_list[j],temp_context_HH,j,True,freqs_resolution,models_dict['entropy_HH'],
                 dec,code_block_size,subband_w[j],subband_h[j],
                 padding_sub_w[j],padding_sub_h[j],yuv_low_bound[3 * j + 3],
                 yuv_high_bound[3 * j + 3],shift_min[:, 3 * j + 3],shift_max[:, 3 * j + 3],min_v[:, 3 * j + 3])
            
            HH_list[j]=band_temp
            coded_coe_num+=coded_num
            
            print('HH' + str(j) + ' decoded')
            
            LL = models_dict['transform'].inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j], j)

        assert (coded_coe_num == (height + pad_h) * (width + pad_w) * 3)

        recon = LL.permute(1, 0, 2, 3)
        recon = recon[:, :, 0:height, 0:width]
        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy()
        recon = yuv2rgb_lossless(recon).astype(np.float32)
        recon = np.clip(recon, 0., 255.).astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(args.recon_dir + '/' + bin_name + '.png')

        logfile.flush()
