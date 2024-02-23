import torch
from torch.nn import functional as F
import Model.PixelCNN_lossless as PixelCNN_lossless
import Model.learn_wavelet_trans_lossless as learn_wavelet_trans


class Model(torch.nn.Module):
    def __init__(self,decomp_levels,alpha=1.0,trainAlpha=True,lifting_struct="2D-NSWT-LCLS",non_linear_arch="Proposed",wavelet="53",is_train=False,only_train_entropy_model=False,nec=128,):
        super(Model, self).__init__()

        self.wavelet = wavelet
        self.decomp_levels = decomp_levels
        self.only_train_entropy_model = only_train_entropy_model
        self.non_linear_arch=non_linear_arch
        self.is_train=is_train
        print(lifting_struct)
        print(non_linear_arch)
        print(wavelet)
        self.wavelet_transform = torch.nn.ModuleList(learn_wavelet_trans.Wavelet_Transform_2D(alpha=alpha,trainAlpha=trainAlpha,lifting_struct=lifting_struct,isTrain=is_train,use_bias=True,use_init=True,wavelet=self.wavelet,non_linear_arch=non_linear_arch) for _i in range(self.decomp_levels))  

        if self.is_train:
            self.coding_LL = PixelCNN_lossless.PixelCNN(nec)
            self.coding_HL_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(1,nec) for _i in range(self.decomp_levels)])
            self.coding_LH_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(2,nec) for _i in range(self.decomp_levels)])
            self.coding_HH_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(3,nec) for _i in range(self.decomp_levels)])
            

    def forward_trans(self, x):
        LL = x
       
        HL_list = []
        LH_list = []
        HH_list = []
    
        for i in range(self.decomp_levels):
            if self.only_train_entropy_model:
                with torch.no_grad():
                    LL, LH, HL, HH = self.wavelet_transform[i].forward_trans(LL)
            else:
                LL, LH, HL, HH = self.wavelet_transform[i].forward_trans(LL)
                
            LH_list.append(LH)
            HL_list.append(HL)
            HH_list.append(HH)
        

        if self.is_train:
            bits = self.coding_LL.forward_train(LL)

            for i in range(self.decomp_levels):

                j = self.decomp_levels - 1 - i
            
                bits = bits + self.coding_HL_list[j].forward_train(HL_list[j],LL)
                bits = bits + self.coding_LH_list[j].forward_train(LH_list[j], torch.cat((LL, HL_list[j]),1))
                bits = bits + self.coding_HH_list[j].forward_train(HH_list[j], torch.cat((LL, HL_list[j], LH_list[j]),1))

                LL = self.inverse_trans(LL, LH_list[j], HL_list[j] , HH_list[j],j)
            
            return bits,LL,LH_list,HL_list,HH_list
        
        return LL,LH_list,HL_list,HH_list
    
    def inverse_trans(self, LL, LH,HL,HH, layer):

        LL = self.wavelet_transform[layer].inverse_trans(LL,LH,HL,HH)

        return LL

class CodingLL_lossless(torch.nn.Module):
    def __init__(self,nec=128):
        super(CodingLL_lossless, self).__init__()

        self.coding_LL = PixelCNN_lossless.PixelCNN(nec)

    def forward(self, LL, lower_bound, upper_bound):
        prob = self.coding_LL.forward_inf(LL, lower_bound, upper_bound)
        return prob


class CodingHL_lossless(torch.nn.Module):
    def __init__(self,decomp_levels,nec=128):
        super(CodingHL_lossless, self).__init__()

        self.decomp_levels = decomp_levels

        self.coding_HL_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(1,nec) for _i in range(self.decomp_levels)])

    def forward(self, HL, context, lower_bound, upper_bound, layer):
        prob = self.coding_HL_list[layer].forward_inf(HL, context, lower_bound, upper_bound)
        return prob


class CodingLH_lossless(torch.nn.Module):
    def __init__(self,decomp_levels,nec=128):
        super(CodingLH_lossless, self).__init__()

        self.decomp_levels = decomp_levels
        self.coding_LH_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(2,nec) for _i in range(self.decomp_levels)])

    def forward(self, LH, context, lower_bound, upper_bound, layer):
        prob = self.coding_LH_list[layer].forward_inf(LH, context, lower_bound, upper_bound)
        return prob


class CodingHH_lossless(torch.nn.Module):
    def __init__(self,decomp_levels,nec=128):
        super(CodingHH_lossless, self).__init__()

        self.decomp_levels = decomp_levels
        self.coding_HH_list = torch.nn.ModuleList([PixelCNN_lossless.PixelCNN_Context(3,nec) for _i in range(self.decomp_levels)])

    def forward(self, HH, context, lower_bound, upper_bound, layer):
        prob = self.coding_HH_list[layer].forward_inf(HH, context, lower_bound, upper_bound)
        return prob
