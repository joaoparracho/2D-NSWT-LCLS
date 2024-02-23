import argparse
import json

from inference.enc_lossless import enc_lossless

parser = argparse.ArgumentParser(description='IEEE 1857.11 FVC CFP', conflict_handler='resolve')
# parameters
parser.add_argument('--cfg_file', type=str, default='D:/liqiang/iwave/iwave_normal/cfg/encode.cfg')

args, unknown = parser.parse_known_args()

cfg_file = args.cfg_file
with open(cfg_file, 'r') as f:
    cfg_dict = json.load(f)
    
    for key, value in cfg_dict.items():
        if isinstance(value, int):
            parser.add_argument('--{}'.format(key), type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument('--{}'.format(key), type=float, default=value)
        else:
            parser.add_argument('--{}'.format(key), type=str, default=value)

cfg_args, unknown = parser.parse_known_args()

parser.add_argument('--input_dir', type=str, default=cfg_args.input_dir)
parser.add_argument('--img_name', type=str, default=cfg_args.img_name)
parser.add_argument('--bin_dir', type=str, default=cfg_args.bin_dir)
parser.add_argument('--log_dir', type=str, default=cfg_args.log_dir)
parser.add_argument('--recon_dir', type=str, default=cfg_args.recon_dir)
parser.add_argument('--isLossless', type=int, default=cfg_args.isLossless)
parser.add_argument('--model_qp', type=int, default=cfg_args.model_qp)
parser.add_argument('--qp_shift', type=int, default=cfg_args.qp_shift)
parser.add_argument('--lifting_struct', type=str, default=cfg_args.lifting_struct)
parser.add_argument('--non_linear_arch', type=str, default=cfg_args.non_linear_arch)
parser.add_argument('--wavelet', type=str, default=cfg_args.wavelet)
parser.add_argument('--decomp_levels', type=int, default=cfg_args.decomp_levels,help="Number of stages")
parser.add_argument('--num_ch_encode', type=int, default=3)

parser.add_argument('--model_path', type=str, default=cfg_args.model_path) # store all models

parser.add_argument('--code_block_size', type=int, default=cfg_args.code_block_size)


def main():
    args = parser.parse_args()

    enc_lossless(args)

if __name__ == "__main__":
    main()
