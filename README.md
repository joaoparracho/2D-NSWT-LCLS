# 2D-NSWT-LCLS
This repo provides the official implementation of "Non-Separable Wavelet Transform using Learnable Convolutional Lifting Steps"

# Usage

To train the model, run `source/train_lossles.py`:

```
python3 source/train_lossless.py --input_dir path/to/train/dataset/ --model_dir path/to/save/model/checkpoints/ --log_dir path/to/save/train/logs/ --epochs 1750 --patch_size 128 --batch_size 2 --decomp_levels 4 --num_workers 8 --only_train_entropy_model 0 --lr 0.001 --alpha 0.1 --nec 128 --lifting_struct 2D-NSWT-LCLS --non_linear_arch Proposed --wavelet 53
```
- For the `--lifting_struct` argument, there are two lifting structures available: `2D-NSWT-LCLS`, and `Generic`
- For the `--non_linear_arch` argument, there are four lifting operators architectures available: `Proposed`, `Varian_A`, `Varian_B`, and `Varian_C`

To encode an image, run `source/encodeAPP.py`:

```
python3 source/encodeAPP.py --cfg_file encode.cfg
```

To decode an image, run `source/decodeAPP.py`:

```
python3 source/decodeAPP.py --cfg_file decode.cfg
```

To run a full pipeline of training the model and then encode and decode the test dataset images, run `launch_train.py`:

```
python3 launch_train.py
```
- Within this script, there is an example on how to generate the config files for both encoding and decoding step

