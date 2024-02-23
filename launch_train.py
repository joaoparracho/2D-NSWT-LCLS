import os
import json
import shutil
import contextlib
import multiprocessing as mp

source_path="/nfs/home/jparracho.it/PlenoIsla/Sources/2D-NSWT-LCLS/"
dataset_path="/nfs/home/jparracho.it/PlenoIsla/Sources/iwave/dataset/"

train_dataset = "DIV2K_train_HR"
test_datasets = ["Kodak"] # Kodak | "DIV2K_train_HR" | "CLIC"
lifting_structs=["2D-NSWT-LCLS"] # 2D-NSWT-LCLS | Generic
non_linear_archs=["Proposed"] # Proposed | Variant_A | Variant_B | Variant_C
wavelets=["53"] # 53 | 97
decomp_levels = ["4"] 
only_train_entropy_models = ["0"]
alpha="0.1"
trainAlpha=True
shuffle=False
necs = ["128"]
epochs = "150"
patch_size = "128"
batches = ["2"]
lr = "0.0001" 
device_train = "0"
device_test = "0"

decode_images=False

for decomp_level in decomp_levels:
    for only_train_entropy_model in only_train_entropy_models:
        for batch in batches:
            for nec in necs:
                for lifting_struct in lifting_structs:
                    for non_linear_arch in non_linear_archs:
                        for wavelet in wavelets:
                            not_trained=True
                            for test_dataset in test_datasets:

                                name = "lossless_"+lifting_struct+"_"+non_linear_arch+"_"+wavelet+"_decomp"+decomp_level+"_b"+batch+"_e"+epochs+"_p"+patch_size+"_trentr"+only_train_entropy_model+"_alpha_"+alpha+"_"+str(int(trainAlpha))+"_lr_"+lr+"_nec"+nec                                
                                
                                cmd = "CUDA_VISIBLE_DEVICES="+device_train+" python3 source/train_lossless.py --input_dir "+dataset_path+train_dataset+"/"+ \
                                " --model_dir "+source_path+"output/"+name+"/"+ \
                                " --test_input_dir "+dataset_path+test_dataset+"/"+ \
                                " --log_dir "+source_path+"output/"+name+"/"+\
                                " --epochs "+epochs+" --patch_size "+patch_size+" --batch_size "+batch+\
                                " --decomp_levels "+decomp_level+" --num_workers 8"+\
                                " --only_train_entropy_model "+only_train_entropy_model+" --lr "+lr+" --alpha "+alpha+" --nec "+nec+\
                                " --lifting_struct "+lifting_struct+" --non_linear_arch "+non_linear_arch+" --wavelet "+wavelet
                                
                                if trainAlpha:
                                    cmd += " --trainAlpha"
                                if shuffle:
                                    cmd += " --shuffle"
                                    
                                # cmd += " --alpha_array"
                                # for i in range(int(decomp_level)):
                                #     for j in range(6):
                                #         cmd += " "+str(alpha_array[i*int(decomp_level) + j])
                                
                                if not_trained:
                                    os.system(cmd)
                                    not_trained=False
                                
                                if not os.path.exists(source_path+"cfg/"+test_dataset+"_"+name):
                                    os.mkdir(source_path+"cfg/"+test_dataset+"_"+name)
                                
                                if not os.path.exists(source_path+"cfg/decode/"):
                                    os.mkdir(source_path+"cfg/decode/")
                                    
                                if not os.path.exists(source_path+"cfg/decode/"+test_dataset+"_"+name):
                                    os.mkdir(source_path+"cfg/decode/"+test_dataset+"_"+name)
                                
                                if not os.path.exists(source_path+"output/"+test_dataset):
                                    os.mkdir(source_path+"output/"+test_dataset)
                                    
                                if not os.path.exists(source_path+"output/"+test_dataset+"/"+name):
                                    os.mkdir(source_path+"output/"+test_dataset+"/"+name)
                                    
                                
                                if not os.path.exists(source_path+"output/"+test_dataset+"/decode/"):
                                    os.mkdir(source_path+"output/"+test_dataset+"/decode/")
                                    
                                if not os.path.exists(source_path+"output/"+test_dataset+"/decode/"+name):
                                    os.mkdir(source_path+"output/"+test_dataset+"/decode/"+name)
                                
                                epochs=10
                                checkpoint = f"model_epoch{int(epochs):03}.pth"
                             
                                shutil.copy(source_path+"output/"+name+"/"+checkpoint,source_path+"output/"+test_dataset+"/"+name+"/9999_lossless.pth")
                                
                                if test_dataset =="Kodak":
                                    test_id_init=1
                                    test_id_last=25
                                elif test_dataset =="DIV2K_train_HR":
                                    test_id_init=801
                                    test_id_last=901
                                elif test_dataset =="CLIC":
                                    test_id_init=1
                                    test_id_last=42
                                
                                for i in range(test_id_init,test_id_last):
                                    if test_dataset =="Kodak":
                                        idx = f"{i:02}"
                                        img_name = "kodim"+idx+".png"
                                        bin_name = "kodim"+idx+"_27_0.bin"
                                    elif test_dataset =="DIV2K_train_HR" or test_dataset =="CLIC":
                                        idx = f"{i:04}"
                                        img_name = idx+".png"
                                        bin_name = idx+"_27_0.bin"
                                        
                                    cfg = {
                                        "input_dir": dataset_path+test_dataset+"/test",
                                        "img_name": img_name,
                                        "bin_dir": source_path+"output/"+test_dataset+"/"+name+"",
                                        "log_dir": source_path+"output/"+test_dataset+"/"+name+"",
                                        "recon_dir": source_path+"output/"+test_dataset+"/"+name+"",
                                        "model_path": source_path+"output/"+test_dataset+"/"+name+"",
                                        "code_block_size": 64,
                                        "isLossless": 1,
                                        "model_qp": 27,
                                        "qp_shift": 0,
                                        "decomp_levels": int(decomp_level),
                                        "lifting_struct":lifting_struct,
                                        "non_linear_arch":non_linear_arch,
                                        "wavelet":wavelet,
                                        "alpha": float(alpha),
                                        "nec":int(nec),
                                    }
                                    
                                    
                                    if test_dataset =="Kodak":
                                        with open("cfg/"+test_dataset+"_"+name+"/encode_kodim"+idx+".cfg", "w") as output_file:
                                            json.dump(cfg, output_file, indent=2)
                                    elif test_dataset =="DIV2K_train_HR" or test_dataset =="CLIC":
                                        with open("cfg/"+test_dataset+"_"+name+"/encode_"+idx+".cfg", "w") as output_file:
                                            json.dump(cfg, output_file, indent=2)
                                        
                                    
                                    cfg = {
                                        "input_dir": dataset_path+test_dataset+test_dataset+"/test",
                                        "img_name": img_name,
                                        "bin_file": source_path+"output/"+test_dataset+"/"+name+"/"+bin_name,
                                        "log_dir": source_path+"output/decode/"+test_dataset+"/"+name+"",
                                        "recon_dir": source_path+"output/decode/"+test_dataset+"/"+name+"",
                                        "isLossless": 1,
                                        "model_qp": 27,
                                        "qp_shift": 0,
                                        "model_path": source_path+"output/"+test_dataset+"/"+name+"",
                                        "code_block_size": 64,
                                        "decomp_levels": int(decomp_level),
                                        "lifting_struct":lifting_struct,
                                        "non_linear_arch":non_linear_arch,
                                        "wavelet":wavelet,
                                        "alpha": float(alpha),
                                        "nec":int(nec),
                                    }
                                    
                                    if test_dataset =="Kodak":
                                        with open(source_path+"cfg/decode/"+test_dataset+"_"+name+"/decode_kodim"+idx+".cfg", "w") as output_file:
                                            json.dump(cfg, output_file, indent=2)
                                        
                                        os.system("CUDA_VISIBLE_DEVICES="+device_test+" python3 source/encodeAPP.py --cfg_file cfg/"+test_dataset+"_"+name+"/encode_kodim"+idx+".cfg")
                                        
                                        if decode_images:
                                            os.system("CUDA_VISIBLE_DEVICES="+device_test+" python3 source/decodeAPP.py --cfg_file cfg/decode/"+test_dataset+"_"+name+"/decode_kodim"+idx+".cfg")
                                        
                                    elif test_dataset =="DIV2K_train_HR" or test_dataset =="CLIC":
                                        with open(source_path+"cfg/decode/"+test_dataset+"_"+name+"/decode_"+idx+".cfg", "w") as output_file:
                                            json.dump(cfg, output_file, indent=2)
                                        
                                        os.system("CUDA_VISIBLE_DEVICES="+device_test+" python3 source/encodeAPP.py --cfg_file cfg/"+test_dataset+"_"+name+"/encode_"+idx+".cfg")
                                        
                                        if decode_images:
                                            os.system("CUDA_VISIBLE_DEVICES="+device_test+" python3 source/decodeAPP.py --cfg_file cfg/decode/"+test_dataset+"_"+name+"/decode_"+idx+".cfg")
                                    
                                    
                                os.system("python3 computeBPP.py --path "+source_path+"output/"+test_dataset+"/"+name)
