from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import argparse
import torch
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt
from Model.Model_lossless import Model
import random
from torch.utils.tensorboard import SummaryWriter
from Util.dataset import DatasetYCoCb

# torch.autograd.set_detect_anomaly(True)
import time

parser = argparse.ArgumentParser(description='.')

# parameters
parser.add_argument('--input_dir', type=str, default="/nfs/home/jparracho.it/PlenoIsla/Sources/iwave/dataset/DIV2K_train_HR/")
parser.add_argument('--test_input_dir', type=str, default="/nfs/home/jparracho.it/PlenoIsla/Sources/iwave/dataset/Kodak/")
parser.add_argument('--val_input_dir', type=str, default="/nfs/data/share/datasets/CLIC_Challenge_2020/")
parser.add_argument('--model_dir', type=str, default="/nfs/home/jparracho.it/PlenoIsla/Sources/iwave/output/lossless")
parser.add_argument('--log_dir', type=str, default="/nfs/home/jparracho.it/PlenoIsla/Sources/iwave/output/")
parser.add_argument('--decomp_levels', type=int, default=1,help="Number of decomposition levels")
parser.add_argument('--only_train_entropy_model', type=int, default=0)
parser.add_argument('--lifting_struct', type=str, default="2D-NSWT-LCLS")
parser.add_argument('--non_linear_arch', type=str, default="Proposed")
parser.add_argument('--wavelet', type=str, default="53")
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--alpha_array', nargs='+', default=[])
parser.add_argument('--trainAlpha', action='store_true')
parser.add_argument('--nec', type=int, default=128,help="Number of channels for entropy module")
parser.add_argument('--epochs', type=int, default=200) 
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_Wn_weigts_png', action='store_true')

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def ycocb2rgb(x):
    
    x = np.array(x, dtype=np.int32)

    Y = x[:, :, :, 0:1]
    Co = x[:, :, :, 1:2]
    Cg = x[:, :, :, 2:3]

    rgb = np.zeros_like(x, dtype=np.int32)

    tmp = Y - np.right_shift(Cg, 1)
    g = Cg + tmp
    b = tmp - np.right_shift(Co, 1)
    r = b + Co

    
    rgb[:, :, :, 0:1] = r
    rgb[:, :, :, 1:2] = g
    rgb[:, :, :, 2:3] = b

    return rgb

def main():
    args = parser.parse_args()
    print(args)
    input_dir = args.input_dir
    test_input_dir = args.test_input_dir
    val_input_dir = args.val_input_dir
    

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    ckpt_dir = args.model_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logfile = open(args.log_dir + '/log.txt', 'w')
    
    if args.save_Wn_weigts_png:
        if not os.path.exists(ckpt_dir+"/Wn_weights/"):
            os.makedirs(ckpt_dir+"/Wn_weights/")

    total_epoch = args.epochs
    patch_size = args.patch_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    only_train_entropy_model = args.only_train_entropy_model
    print(only_train_entropy_model)
    
    writer = SummaryWriter("runs/"+args.model_dir.split("/")[-2])

    is_train=True
    model = Model(args.decomp_levels,args.alpha,args.trainAlpha,args.lifting_struct,args.non_linear_arch,args.wavelet,is_train,args.only_train_entropy_model,args.nec,)
                 
    if len(args.alpha_array)>0:
        for i in range(len(model.wavelet_transform)):
            j=0
            print(str(i+1)+" Level of decomposition")
            for name, layer in model.wavelet_transform[i].lifting.named_modules(): 
                if name.__contains__("residual") and not name.__contains__("."): 
                    layer.alpha = torch.nn.Parameter(torch.Tensor([float(args.alpha_array[i*4+j])]),requires_grad=False)
                    j+=1

    epoch = 1
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        epoch = checkpoint["epoch"] + 1
        state_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('Load pre-trained model [' + args.load_model + '] succeed!')
        logfile.write('Load pre-trained model [' + args.load_model + '] succeed!' + '\n')
        logfile.flush()

    for i in range(len(model.wavelet_transform)):
        print(str(i+1)+" Level of decomposition")
        for name, layer in model.wavelet_transform[i].lifting.named_modules(): 
            if hasattr(layer, 'alpha'):
                print(name+"_alpha: "+str(round(layer.alpha.item(),4)))
            
    if torch.cuda.is_available():
        model = model.cuda()

    model.train()

    logfile.write('Load data starting...' + '\n')
    logfile.flush()
    print('Load data starting...')
   
    shuffle=args.shuffle
    
    train_data=DatasetYCoCb(input_dir, split="train",transform=transforms.Compose([transforms.RandomCrop(patch_size)]))
    val_data=DatasetYCoCb(val_input_dir, split="valid",transform=transforms.Compose([transforms.CenterCrop(patch_size)]))
    test_data=DatasetYCoCb(test_input_dir, split="test",transform=transforms.Compose([transforms.RandomCrop(patch_size)]))
    
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader=DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    logfile.write('Load all data succeed!' + '\n')
    logfile.flush()
    print('Load all data succeed!')
    
    max_step = train_loader.__len__()
    max_step = 20000 if 20000 < max_step else max_step
    print("train step:", max_step, end=" ")
    logfile.write("train step:" + str(max_step))
    test_max_step = test_loader.__len__()
    test_max_step = 1000 if 1000 < test_max_step else test_max_step
    print("test step:", test_max_step)
    logfile.write(" test step:" + str(test_max_step) + '\n')
    val_max_step = val_loader.__len__()
    val_max_step = 1000 if 1000 < val_max_step else val_max_step

    print("LR: "+str(args.lr))
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    while True:
        if epoch > total_epoch:
            break

        bpp_all = 0.
        start_time = time.time()

        for batch_idx, data in enumerate(train_loader): 
            
            if batch_idx > max_step - 1:
                break
            
            input_img_v = to_variable(data["data"])
            ori_img = data["data_rgb"].cpu().data.numpy()
            
            size = input_img_v.size()
            input_img_v = input_img_v.view(-1, 1, size[2], size[3])
            

            bits, recon_img,LH_list, HL_list,HH_list = model.forward_trans(input_img_v)
            
            
            bpp = torch.sum(bits)/size[0]/size[2]/size[3]

            recon_img = recon_img.view(-1, 3, size[2], size[3])
            recon_img = recon_img.permute(0,2,3,1).cpu().data.numpy()
            recon_img = ycocb2rgb(recon_img).astype(np.float32)
            mse = np.mean((ori_img - recon_img)**2)
            
            psnr = 10. * np.log10(255. * 255. / mse) if mse > 0.0 else 100

            opt.zero_grad()
            
            bpp.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()
            
            bpp_all += bpp.item()

            if batch_idx % 10 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                                + 'train loss: ' + str(bpp.item())+ '/'  + str(psnr) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                        + 'train loss: ' + str(bpp.item())+ '/'  + str(psnr))
          
        for i in range(len(model.wavelet_transform)):
            print(str(i+1)+" Level of decomposition")
            for name, layer in model.wavelet_transform[i].lifting.named_modules(): 
                if hasattr(layer, 'alpha'):
                    print(name+"_alpha: "+str(round(layer.alpha.item(),4)))

        print("Epoch: "+str(time.time()-start_time) +" s")
       
        bpp_all_test = 0.
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):

                if batch_idx > test_max_step - 1:
                    break

                input_img_v = to_variable(data["data"])
                ori_img = data["data_rgb"].cpu().data.numpy()
                
                size = input_img_v.size()
                input_img_v = input_img_v.view(-1, 1, size[2], size[3])

                bits, recon_img,LH_list,HL_list,HH_list = model.forward_trans(input_img_v)

                bpp = torch.sum(bits)/size[0]/size[2]/size[3]

                recon_img = recon_img.view(-1, 3, size[2], size[3])
                recon_img = recon_img.permute(0,2,3,1).cpu().data.numpy()
                recon_img = ycocb2rgb(recon_img).astype(np.float32)
                mse = np.mean((ori_img - recon_img)**2)
                
                psnr = 10. * np.log10(255. * 255. / mse) if mse > 0.0 else 100

                bpp_all_test += bpp.item()

                if batch_idx % 10 == 0:
                    logfile.write('Test Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(test_max_step) + ']   '
                                    + 'test loss: ' + str(bpp.item())+ '/'  + str(psnr) + '\n')
                    logfile.flush()
                    print('Test Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(test_max_step) + ']   '
                            + 'test loss: ' + str(bpp.item())+ '/'  + str(psnr))

        bpp_all_val = 0.
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):

                if batch_idx > val_max_step - 1:
                    break

                input_img_v = to_variable(data["data"])
                ori_img = data["data_rgb"].cpu().data.numpy()
                
                size = input_img_v.size()
                input_img_v = input_img_v.view(-1, 1, size[2], size[3])

                bits, recon_img,LH_list,HL_list,HH_list = model.forward_trans(input_img_v)
                
     
                bpp = torch.sum(bits)/size[0]/size[2]/size[3]

                recon_img = recon_img.view(-1, 3, size[2], size[3])
                recon_img = recon_img.permute(0,2,3,1).cpu().data.numpy()
                recon_img = ycocb2rgb(recon_img).astype(np.float32)
                mse = np.mean((ori_img - recon_img)**2)
                
                psnr = 10. * np.log10(255. * 255. / mse) if mse > 0.0 else 100

                bpp_all_val += bpp.item()

                if batch_idx % 10 == 0:
                    logfile.write('Val Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                                    + 'val loss: ' + str(bpp.item())+ '/'  + str(psnr) + '\n')
                    logfile.flush()
                    print('Val Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                            + 'val loss: ' + str(bpp.item())+ '/'  + str(psnr))

        if epoch % 50 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '.pth', _use_new_zipfile_serialization=False)

        
        
        
        logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
        logfile.flush()
        print('bpp_mean: ' + str(bpp_all / max_step))
        print('bpp_test_mean: ' + str(bpp_all_test / test_max_step))
        print('bpp_val_mean: ' + str(bpp_all_val / val_max_step))
        
        losses = {"bpp_train":bpp_all / max_step,"bpp_test":bpp_all_test / test_max_step,"bpp_val":bpp_all_val / val_max_step}
        writer.add_scalars(args.model_dir.split("/")[-2],losses, epoch)
        epoch  = epoch + 1

    logfile.close()

if __name__ == "__main__":
    main()
