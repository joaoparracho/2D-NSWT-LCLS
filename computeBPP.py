import os
import glob
import argparse
from PIL import Image
import numpy as np 


parser = argparse.ArgumentParser(description='.')
parser.add_argument('--path', type=str, default="/nfs/home/jparracho.it/PlenoIsla/Sources/iwave/output/Kodak/lossless_Non_Seperable_stage1_gp1_b24_e150_p128_lbasic_winit_trntw1")


bpp = 0
bpp_2 = 0
args = parser.parse_args()
i=0
for bin_path in glob.glob(args.path+"/*.bin"):
    file=bin_path.split(".bin")[0]+".png"
    img = Image.open(file).convert("RGB")
    img = np.array(img, dtype=np.float32)
    print(str(img.shape[0])+"x"+str(img.shape[1]))
    bitstream_size=os.path.getsize(bin_path)
    bpp += (bitstream_size*8)/(img.shape[0]*img.shape[1])
    i = i+1

logfile = open(args.path + "/bpp.txt", 'w')

logfile.write(args.path+"\n")
logfile.write(str(bpp/i))

print(args.path)
print(bpp/i)
print(i)

