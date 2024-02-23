# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from pathlib import Path

from PIL import Image
import numpy as np
import  torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import random
import glob

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def rgb2ycocb(x):
    x = np.array(x, dtype=np.int32)

    r = x[ :, :, 0:1]
    g = x[ :, :, 1:2]
    b = x[ :, :, 2:3]

    yuv = np.zeros_like(x, dtype=np.int32)

    Co = r - b
    tmp = b + np.right_shift(Co, 1)
    Cg = g - tmp
    Y = tmp + np.right_shift(Cg, 1)

    yuv[ :, :, 0:1] = Y
    yuv[ :, :, 1:2] = Co
    yuv[ :, :, 2:3] = Cg

    return yuv

def ycocb2rgb(x):
    
    x = np.array(x, dtype=np.int32)

    Y =  x[:, :, :, 0:1]
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


class DatasetYCoCb(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        self.img_path = None

        self.samples = []
        
        splitdir = Path(root) / split
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{splitdir}"')
        self.samples.extend([f for f in splitdir.iterdir() if f.is_file()])

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        img=self.transform(img)
        self.img_path = self.samples[index]
        img = np.array(img, dtype=np.float32)
        img_rgb = copy.deepcopy(img)

        ycbcr = rgb2ycocb(img).astype(np.float32)
        ycbcr_t = torch.from_numpy(ycbcr).permute(2,0,1)
        return {"data": ycbcr_t, "data_rgb": img_rgb,"id":self.samples[index]._str}
     


    def __len__(self):
        return len(self.samples)