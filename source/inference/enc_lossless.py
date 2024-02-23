import copy
import os
import time

import numpy as np
import random
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F


import inference.arithmetic_coding as ac
import Model.Model_lossless as Model
from inference.utils import (find_min_and_max, img2patch, img2patch_padding,
                   model_lambdas, qp_shifts, rgb2yuv_lossless,
                   yuv2rgb_lossless)

import socket



def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def write_binary(enc, value, bin_num):
    bin_v = '{0:b}'.format(value).zfill(bin_num)
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        enc.write(freqs, int(bin_v[i]))


def enc_subband(subband,temp_context,decomp_level,use_context,entropy_model,enc,code_block_size,subband_w,subband_h,padding_sub_w,padding_sub_h,yuv_low_bound,yuv_high_bound,freqs_resolution,shift_min,shift_max,min_v,num_ch_encode=3):
    
    coded_coe_num = 0
    tmp_stride = subband_w + padding_sub_w
    tmp_hor_num = tmp_stride // code_block_size
    paddings = (0, padding_sub_w, 0, padding_sub_h)
    enc_band = F.pad(subband, paddings, "constant")
    enc_band = img2patch(enc_band, code_block_size, code_block_size, code_block_size)
    
    if use_context:
        if padding_sub_w >= temp_context.shape[3] or padding_sub_h >= temp_context.shape[2]:
            context = F.pad(temp_context, paddings, "constant")
        else:
            context = F.pad(temp_context, paddings, "reflect")
        
    paddings = (6, 6, 6, 6)
    enc_band = F.pad(enc_band, paddings, "constant")
    if use_context:
        context = F.pad(context, paddings, "reflect")
        context = img2patch_padding(context, code_block_size+12, code_block_size+12, code_block_size, 6)
    
    for h_i in range(code_block_size):
        for w_i in range(code_block_size):
            cur_ct = copy.deepcopy(enc_band[:, :, h_i:h_i + 13, w_i:w_i + 13])
            cur_ct[:, :, 13 // 2 + 1:13, :] = 0.
            cur_ct[:, :, 13 // 2, 13 // 2:13] = 0.
            
            if use_context:
                cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                prob = entropy_model(cur_ct, cur_context,yuv_low_bound, yuv_high_bound, decomp_level)
            else:
                prob = entropy_model(cur_ct,yuv_low_bound,yuv_high_bound)

            prob = prob.cpu().data.numpy()
            index = enc_band[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int)

            prob = prob * freqs_resolution
            prob = prob.astype(np.int64)

            for sample_idx, prob_sample in enumerate(prob):
                coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size*code_block_size + \
                            h_i * tmp_stride + \
                            ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                            w_i
                if (coe_id % tmp_stride) < subband_w and (coe_id // tmp_stride) < subband_h:
                    yuv_flag = sample_idx % 3
                    
                    if shift_min[yuv_flag] < shift_max[yuv_flag]  and (yuv_flag<num_ch_encode):
                        freqs = ac.SimpleFrequencyTable(prob_sample[shift_min[yuv_flag]:shift_max[yuv_flag] + 1])
                        data = index[sample_idx] - min_v[yuv_flag]
                        assert data >= 0
                        enc.write(freqs, data)
                        coded_coe_num = coded_coe_num + 1
    return coded_coe_num
                            
def enc_lossless(args):
    
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

    assert args.isLossless==1

    if not os.path.exists(args.bin_dir):
        os.makedirs(args.bin_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.recon_dir):
        os.makedirs(args.recon_dir)

    logfile = open(args.log_dir + '/enc_log.txt', 'a')

    assert args.model_qp==27
    assert args.qp_shift==0
    init_scale = qp_shifts[args.model_qp][args.qp_shift]
    print(init_scale)
    logfile.write(str(init_scale) + '\n')
    logfile.flush()

    code_block_size = args.code_block_size

    bin_name = args.img_name[0:-4] + '_' + str(args.model_qp) + '_' + str(args.qp_shift)
    
    print(socket.gethostname())
    bit_out = ac.CountingBitOutputStream(bit_out=ac.BitOutputStream(open(args.bin_dir + '/' + bin_name +'.bin', "wb")))
    enc = ac.ArithmeticEncoder(bit_out)

    freqs_resolution = 1e6

    freqs = ac.SimpleFrequencyTable(np.ones([2], dtype=np.int))
    enc.write(freqs, args.isLossless)

    decomp_levels = args.decomp_levels

    model_path = args.model_path + '/' + str(model_lambdas[args.model_qp]) + '_lossless.pth'
    
    print(model_path)
    checkpoint = torch.load(model_path)
    
    all_part_dict = checkpoint['state_dict']

    models_dict = {}

    print("ENCODING_LOSSLESS")
    
    args.nec=128
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

    img_path = args.input_dir + '/' + args.img_name

    with torch.no_grad():
        start = time.time()

        print(img_path)
        logfile.write(img_path + '\n')
        logfile.flush()

        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)
        original_img = copy.deepcopy(img)

        img = rgb2yuv_lossless(img).astype(np.float32)
        img = torch.from_numpy(img)
        # img -> [n,c,h,w]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        # original_img = img

        # img -> (%16 == 0)
        size = img.size()
        height = size[2]
        width = size[3]
        # encode height and width, in the range of [0, 2^15=32768]
        write_binary(enc, height, 15)
        write_binary(enc, width, 15)

        pad_h = int(np.ceil(height / (2**(decomp_levels)))) * (2**(decomp_levels)) - height
        pad_w = int(np.ceil(width / (2**(decomp_levels)))) * (2**(decomp_levels)) - width
        
        paddings = (0, pad_w, 0, pad_h)
        img = F.pad(img, paddings, 'replicate')

        # img -> [3,1,h,w], YUV in batch dim
        # img = rgb2yuv(img)
        img = img.permute(1, 0, 2, 3)

        input_img_v = to_variable(img)
        
        LL, LH_list, HL_list, HH_list = models_dict['transform'].forward_trans(input_img_v)

        min_v, max_v = find_min_and_max(LL, HL_list, LH_list, HH_list,decomp_levels)
        # for all models, the quantized coefficients are in the range of [-6016, 12032]
        # 15 bits to encode this range
        for i in range(3): #3
            for j in range(decomp_levels*3+1): #decomp_levels*3+1 -> 13
                
                tmp = min_v[i, j] + 6016
                write_binary(enc, tmp, 15)
                tmp = max_v[i, j] + 6016
                write_binary(enc, tmp, 15)
                
        yuv_low_bound = min_v.min(axis=0)
        yuv_high_bound = max_v.max(axis=0)
        shift_min = min_v - yuv_low_bound
        shift_max = max_v - yuv_low_bound

        subband_h = [ (height + pad_h) // 2**(i+1) for i in range(decomp_levels) ] #  [(height + pad_h) // 2, (height + pad_h) // 4, (height + pad_h) // 8, (height + pad_h) // 16]
        subband_w = [ (width + pad_w) // 2**(i+1) for i in range(decomp_levels) ]  #  [(width + pad_w) // 2, (width + pad_w) // 4, (width + pad_w) // 8, (width + pad_w) // 16]

        padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_h]
        padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_w]

        coded_coe_num = 0

        #compress LL
                            
        coded_coe_num += enc_subband(LL,torch.zeros(1),decomp_levels-1,False,models_dict['entropy_LL'],
                                    enc,code_block_size,subband_w[decomp_levels-1],subband_h[decomp_levels-1],
                                    padding_sub_w[decomp_levels-1],padding_sub_h[decomp_levels-1],yuv_low_bound[0],
                                    yuv_high_bound[0],freqs_resolution,shift_min[:, 0],shift_max[:,0],min_v[:,0],num_ch_encode=args.num_ch_encode)
        print('LL encoded...')

        for i in range(decomp_levels):
            j = decomp_levels - 1 - i


            temp_context_HL = LL
            temp_context_LH = torch.cat((LL, HL_list[j]), dim=1)
            temp_context_HH = torch.cat((LL, HL_list[j], LH_list[j]), dim=1)  
                
            coded_coe_num += enc_subband(HL_list[j],temp_context_HL,j,True,models_dict['entropy_HL'],
                                    enc,code_block_size,subband_w[j],subband_h[j],
                                    padding_sub_w[j],padding_sub_h[j],yuv_low_bound[3 * j + 1],
                                    yuv_high_bound[3 * j + 1],freqs_resolution,shift_min[:, 3 * j + 1],shift_max[:,3 * j + 1],min_v[:,3 * j + 1],num_ch_encode=args.num_ch_encode)

            print('HL' + str(j) + ' encoded...')
            
            coded_coe_num += enc_subband(LH_list[j],temp_context_LH,j,True,models_dict['entropy_LH'],
                                    enc,code_block_size,subband_w[j],subband_h[j],
                                    padding_sub_w[j],padding_sub_h[j],yuv_low_bound[3 * j + 2],
                                    yuv_high_bound[3 * j + 2],freqs_resolution,shift_min[:, 3 * j + 2],shift_max[:,3 * j + 2],min_v[:,3 * j + 2],num_ch_encode=args.num_ch_encode)
            print('LH' + str(j) + ' encoded...')
            
            coded_coe_num += enc_subband(HH_list[j],temp_context_HH,j,True,models_dict['entropy_HH'],
                                    enc,code_block_size,subband_w[j],subband_h[j],
                                    padding_sub_w[j],padding_sub_h[j],yuv_low_bound[3 * j + 3],
                                    yuv_high_bound[3 * j + 3],freqs_resolution,shift_min[:, 3 * j + 3],shift_max[:,3 * j + 3],min_v[:,3 * j + 3],num_ch_encode=args.num_ch_encode)
            
            print('HH' + str(j) + ' encoded...')

            LL = models_dict['transform'].inverse_trans(LL, LH_list[j], HL_list[j], HH_list[j], j)

        assert (coded_coe_num == (height + pad_h) * (width + pad_w) * args.num_ch_encode)

        recon = LL.permute(1, 0, 2, 3)
        recon = recon[:, :, 0:height, 0:width]
        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy()
        recon = yuv2rgb_lossless(recon).astype(np.float32)

        mse = np.mean((recon - original_img) ** 2)
        psnr = (10. * np.log10(255. * 255. / mse))


        recon = np.clip(recon, 0., 255.).astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(args.recon_dir + '/' + bin_name + '.png')

        enc.finish()
        print('encoding finished!')
        logfile.write('encoding finished!' + '\n')
        end = time.time()
        print('Encoding-time: ', end - start)
        logfile.write('Encoding-time: ' + str(end - start) + '\n')

        bit_out.close()
        print('bit_out closed!')
        logfile.write('bit_out closed!' + '\n')

        filesize = bit_out.num_bits / height / width
        print('BPP: ', filesize)
        logfile.write('BPP: ' + str(filesize) + '\n')
        logfile.flush()

        print('PSNR: ', psnr)
        logfile.write('PSNR: ' + str(psnr) + '\n')
        logfile.flush()


    logfile.close()
