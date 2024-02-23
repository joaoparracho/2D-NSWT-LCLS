import torch
from torch.nn import functional as F
from typing import Any
from torch import Tensor

lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781, 1.149604398860241] # bior4.4


class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass: round the input and store it for the backward pass
        rounded = torch.floor(input)
        ctx.save_for_backward(rounded)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass: gradient of the output with respect to the input
        rounded, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Clone the gradient to modify it
        return grad_input

# Define a custom rounding layer using the autograd function
class RoundLayer(torch.nn.Module):
    def forward(self, input):
        return RoundFunction.apply(input)

class FilterMaskedConv2d_init(torch.nn.Conv2d):
    def __init__(self,mask:Tensor,weight_init:Tensor,use_init:bool, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.mask = mask
        
        if use_init:
            self.weight.data = weight_init.to(self.weight.data.dtype)
            self.bias.data = self.bias.data*0
        
    def forward(self, x: Tensor) -> Tensor:
        if self.mask.device != self.weight.device:
            self.mask = self.mask.to(self.weight.device)
        self.weight.data =  self.weight.data * self.mask
        return super().forward(x)
    
def group_bands(x,LL,LH,HL,HH,s_h,s_w,padding=True):
    
    x_predicted = torch.zeros(LL.shape[0],LL.shape[1],s_h,s_w,dtype=LL.dtype,device=LL.device) if x.shape[0]==LL.shape[0] else torch.zeros(x,dtype=x.dtype)

    x_predicted[:,:,0:s_h:2,0:s_w:2] = LL
    x_predicted[:,:,1:s_h:2,0:s_w:2] = LH 
    x_predicted[:,:,0:s_h:2,1:s_w:2] = HL 
    x_predicted[:,:,1:s_h:2,1:s_w:2] = HH 
    
    
    if padding:
        x_predicted_pad=torch.cat((torch.cat((x_predicted[:,:,:,3:4],x_predicted),3),x_predicted[:,:,:,-4:-3]),3)
        x_predicted_pad=torch.cat((torch.cat((x_predicted_pad[:,:,3:4,:],x_predicted_pad),2),x_predicted_pad[:,:,-4:-3,:]),2)
    else:
        x_predicted_pad=x_predicted
        
    return x_predicted_pad

def get_Wavelet_coefficients(wavelet="53",stage=1):
    
    if wavelet=="53":
        p = -0.5
        u = 0.25
        
    if wavelet=="97":
        if stage==1:
            p = -1.58613432
            u = -0.05298011
        if stage==2:
            p = 0.88291107
            u = 0.44350685    

    P1 = torch.tensor([[[[0.0,0.0,0.0,0.0],
                        [0.0,p**2,p,p**2],
                        [0.0,p,0.0,p],
                        [0.0,p**2,p,p**2],]]])

    P2 = torch.tensor([[[[0.0,0.0,0.0,0.0],
                            [0.0,p,0.0,0.0],
                            [0.0,0.0,0.0,0.0],
                            [0.0,p,0.0,0.0],]]])
    
    P3 = torch.tensor([[[[0.0,0.0,0.0,0.0],
                        [0.0,p,0.0,p],
                        [0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0],]]])

    U1 = torch.tensor([[[[u**2,u,u**2,0.0],
                            [u,0.0,u,0.0],
                            [u**2,u,u**2,0.0],
                            [0.0,0.0,0.0,0.0],]]])

    U2 = torch.tensor([[[[0.0,0.0,u,0.0],
                            [0.0,0.0,0.0,0.0],
                            [0.0,0.0,u,0.0],
                            [0.0,0.0,0.0,0.0],]]])
    

    U3 = torch.tensor([[[[0.0,0.0,0.0,0.0],
                            [0.0,0.0,0.0,0.0],
                            [u,0.0,u,0.0],
                            [0.0,0.0,0.0,0.0],]]])
        
    return P1,P2,P3,U1,U2,U3

def get_lifting_operators_mask(wavelet="53",stage=1):
    P1_winit, P2_winit, P3_winit, U1_winit, U2_winit, U3_winit = get_Wavelet_coefficients(wavelet,stage=stage)
    
    mask_P1 = P1_winit.clone()
    mask_P2 = P2_winit.clone()
    mask_P3 = P3_winit.clone()
    mask_U1 = U1_winit.clone()
    mask_U2 = U2_winit.clone()
    mask_U3 = U3_winit.clone()
        
    mask_P1[P1_winit!=0]=1
    mask_P2[P2_winit!=0]=1
    mask_P3[P3_winit!=0]=1
    mask_U1[U1_winit!=0]=1
    mask_U2[U2_winit!=0]=1
    mask_U3[U3_winit!=0]=1
    
    return mask_P1, mask_P2,mask_P3,mask_U1,mask_U2,mask_U3

def init_linear_cnn(lifting_struct="2D-NSWT-LCLS",wavelet="53",stage=1,use_bias=True,use_init=True):
    
    P1_winit, P2_winit, P3_winit, U1_winit, U2_winit, U3_winit = get_Wavelet_coefficients(wavelet,stage=stage)
        
    mask_P1 = P1_winit.clone()
    mask_P2 = P2_winit.clone()
    mask_P3 = P3_winit.clone()
    mask_U1 = U1_winit.clone()
    mask_U2 = U2_winit.clone()
    mask_U3 = U3_winit.clone()
    
    mask_P1[P1_winit!=0]=1
    mask_P2[P2_winit!=0]=1
    mask_P3[P3_winit!=0]=1
    mask_U1[U1_winit!=0]=1
    mask_U2[U2_winit!=0]=1
    mask_U3[U3_winit!=0]=1
    
    if lifting_struct == "Generic":
        mask_P2[:,:,2,0]=1;mask_P2[:,:,1,0]=1;mask_P2[:,:,1,2]=1;mask_P2[:,:,3,0]=1;mask_P2[:,:,2,2]=1;mask_P2[:,:,3,2]=1
        mask_P3[:,:,0,1]=1;mask_P3[:,:,0,2]=1;mask_P3[:,:,0,3]=1;mask_P3[:,:,2,1]=1;mask_P3[:,:,2,2]=1;mask_P3[:,:,2,3]=1
        mask_U2[:,:,0,1]=1;mask_U2[:,:,1,1]=1;mask_U2[:,:,2,1]=1;mask_U2[:,:,0,1]=1;mask_U2[:,:,1,1]=1;mask_U2[:,:,2,3]=1
        mask_U3[:,:,1,0]=1;mask_U3[:,:,1,1]=1;mask_U3[:,:,1,2]=1;mask_U3[:,:,3,0]=1;mask_U3[:,:,3,1]=1;mask_U3[:,:,3,2]=1
                    
    P1_cnn=FilterMaskedConv2d_init(mask_P1, P1_winit, use_init, 1, 1, mask_P1.shape[-1], stride=2, bias=use_bias)
    P2_cnn=FilterMaskedConv2d_init(mask_P2, P2_winit, use_init, 1, 1, mask_P2.shape[-1], stride=2, bias=use_bias)
    P3_cnn=FilterMaskedConv2d_init(mask_P3, P3_winit, use_init, 1, 1, mask_P3.shape[-1], stride=2, bias=use_bias)
    U1_cnn=FilterMaskedConv2d_init(mask_U1, U1_winit, use_init, 1, 1, mask_U1.shape[-1], stride=2, bias=use_bias)
    U2_cnn=FilterMaskedConv2d_init(mask_U2, U2_winit, use_init, 1, 1, mask_U2.shape[-1], stride=2, bias=use_bias)
    U3_cnn=FilterMaskedConv2d_init(mask_U3, U3_winit, use_init, 1, 1, mask_U3.shape[-1], stride=2, bias=use_bias)
    
    return P1_cnn,P2_cnn,P3_cnn,U1_cnn,U2_cnn,U3_cnn


class LCLS_Parallel(torch.nn.Module):
    def __init__(self,alpha=0.1,trainAlpha=True,mask=[]):
        super(LCLS_Parallel, self).__init__()
        
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]),requires_grad=trainAlpha)
        self.d = float(2**8)
        self.tanh = torch.nn.Tanh()
        self.padding = torch.nn.ReflectionPad2d(1)
        
        self.conv_wavelet = FilterMaskedConv2d_init(mask, False, False, 1, 1, mask.shape[-1], stride=2, bias=True)
        
        
    def forward(self, x):
        out = self.conv_wavelet(x/self.d)
        out = self.tanh(out)
        out = out*self.d*self.alpha
        return out

class Variant_A(torch.nn.Module):
    def __init__(self,alpha=0.1,trainAlpha=True):
        super(Variant_A, self).__init__()
        
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]),requires_grad=trainAlpha)
        self.d = float(2**8)
        self.tanh = torch.nn.Tanh()
        self.padding = torch.nn.ReflectionPad2d(1)
        
        self.conv_0 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        
        
    def forward(self, x):
        out = self.conv_0(self.padding(x/self.d))
        out = self.tanh(out)*self.d*self.alpha
        return out+x
    
class Variant_B(Variant_A):
    def __init__(self,alpha=0.1,trainAlpha=True):
        super(Variant_B, self).__init__(alpha=alpha,trainAlpha=trainAlpha)
        
        self.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0)
        
        
    def forward(self, x):
        x_1 = self.conv0(self.padding(x/self.d))
        x_2 = self.tanh(x_1)
        x_3 = self.conv1(self.padding(x_2))
        x_4 = self.tanh(x_3)
        x_5 = self.conv2(self.padding(x_4))
        x_6 = x_3 + x_5
        out = self.conv3(self.padding(x_6))
        out = out*self.d*self.alpha
        
        return out+x
   
class Variant_C(LCLS_Parallel):
    def __init__(self,alpha=0.1,trainAlpha=True,mask=[]):
        super(Variant_C, self).__init__(alpha=alpha,trainAlpha=trainAlpha,mask=mask)
        
        self.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0)
        
    def forward(self, x):
        
        x_0 = self.conv_wavelet(x/self.d)
        x_1 = self.conv0(self.padding(x_0))
        x_2 = self.tanh(x_1)
        x_3 = self.conv1(self.padding(x_2))
        x_4 = self.tanh(x_3)
        x_5 = self.conv2(self.padding(x_4))
        x_6 = x_5 + x_3
        out = self.conv3(self.padding(x_6))
        out = out+x_0
        out = out*self.d*self.alpha
        
        return out

class Non_Separable_2D_WT(torch.nn.Module):
    def __init__(self,lifting_struct="2D-NSWT-LCLS",isTrain=True,use_bias=False,use_init=True,wavelet="53"):
        super(Non_Separable_2D_WT,self).__init__()
        
        self.lifting_struct = lifting_struct
        self.wavelet = wavelet
        self.isTrain = isTrain
        self.use_bias = use_bias
        
        if wavelet == "53":
            self.P1_cnn_1s,  self.P2_cnn_1s,  self.P3_cnn_1s,\
            self.U1_cnn_1s,  self.U2_cnn_1s, self.U3_cnn_1s = init_linear_cnn(lifting_struct=lifting_struct,wavelet=wavelet,stage=1,use_bias=use_bias,use_init=use_init)
        
        if wavelet == "97":
            self.P1_cnn_1s, self.P2_cnn_1s, self.P3_cnn_1s,\
            self.U1_cnn_1s, self.U2_cnn_1s, self.U3_cnn_1s = init_linear_cnn(lifting_struct=lifting_struct,wavelet=wavelet,stage=1,use_bias=use_bias,use_init=use_init)
            self.P1_cnn_2s, self.P2_cnn_2s, self.P3_cnn_2s,\
            self.U1_cnn_2s, self.U2_cnn_2s, self.U3_cnn_2s = init_linear_cnn(lifting_struct=lifting_struct,wavelet=wavelet,stage=2,use_bias=use_bias,use_init=use_init)
        
        self.round_layer = RoundLayer() if isTrain else torch.floor
        
    def forward_trans(self, x):
    
        s_h = x.shape[2]
        s_w = x.shape[3]
        
        x_Img_pad=torch.cat((torch.cat((x[:,:,:,3:4],x),3),x[:,:,:,-4:-3]),3)
        x_Img_pad=torch.cat((torch.cat((x_Img_pad[:,:,3:4,:],x_Img_pad),2),x_Img_pad[:,:,-4:-3,:]),2)
        
        #batch,channel,row,collumn
        LL = x[:,:,0:s_h:2,0:s_w:2] # (2m   , 2n  )
        LH = x[:,:,1:s_h:2,0:s_w:2] # (2m+1 , 2n  )
        HL = x[:,:,0:s_h:2,1:s_w:2] # (2m   , 2n+1)
        HH = x[:,:,1:s_h:2,1:s_w:2] # (2m+1 , 2n+1)
        
        HH = HH + self.round_layer(self.P1_cnn_1s(x_Img_pad))
        
        if self.lifting_struct == "2D-NSWT-LCLS":
            LH = LH + self.round_layer(self.P2_cnn_1s(x_Img_pad))
            HL = HL + self.round_layer(self.P3_cnn_1s(x_Img_pad))
            
        elif self.lifting_struct == "Generic":
            x_p1_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LH = LH + self.round_layer(self.P2_cnn_1s(x_p1_pad))
            x_p2_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HL = HL + self.round_layer(self.P3_cnn_1s(x_p2_pad))
        
        x_predicted_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
        
        LL = LL + self.round_layer(self.U1_cnn_1s(x_predicted_pad))
        
        if self.lifting_struct == "2D-NSWT-LCLS":
            HL = HL + self.round_layer(self.U2_cnn_1s(x_predicted_pad))
            LH = LH + self.round_layer(self.U3_cnn_1s(x_predicted_pad))
            
        elif self.lifting_struct == "Generic":
            x_u1_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HL = HL + self.round_layer(self.U2_cnn_1s(x_u1_pad))
            x_u2_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LH = LH + self.round_layer(self.U3_cnn_1s(x_u2_pad))
        
        if self.wavelet == "97":
            
            x_updated_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HH = HH + self.round_layer(self.P1_cnn_2s(x_updated_pad))
            
            if self.lifting_struct == "2D-NSWT-LCLS":
                LH = LH + self.round_layer(self.P2_cnn_2s(x_updated_pad))
                HL = HL + self.round_layer(self.P3_cnn_2s(x_updated_pad))
                
            elif self.lifting_struct == "Generic":
                x_p1_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                LH = LH + self.round_layer(self.P2_cnn_2s(x_p1_pad_2s))
                x_p2_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                HL = HL + self.round_layer(self.P3_cnn_2s(x_p2_pad_2s))
            
            x_predicted_pad_2 = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LL = LL + self.round_layer(self.U1_cnn_2s(x_predicted_pad_2))
            
            if self.lifting_struct == "2D-NSWT-LCLS":
                HL = HL + self.round_layer(self.U2_cnn_2s(x_predicted_pad_2))
                LH = LH + self.round_layer(self.U3_cnn_2s(x_predicted_pad_2))
            elif self.lifting_struct == "Generic":
                x_u1_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                HL = HL + self.round_layer(self.U2_cnn_2s(x_u1_pad_2s))
                x_u2_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                LH = LH + self.round_layer(self.U3_cnn_2s(x_u2_pad_2s))
        
        return LL, LH, HL, HH
    
    def inverse_trans(self, LL, LH, HL,HH):
        
        s_h = LL.shape[2]*2
        s_w = LL.shape[3]*2
        
        if self.wavelet == "97":
            x_bands_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            LH = LH - self.round_layer(self.U3_cnn_2s(x_bands_pad_2s))
            x_bands_1_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            HL = HL - self.round_layer(self.U2_cnn_2s(x_bands_1_pad_2s))
            x_bands_2_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            LL = LL - self.round_layer(self.U1_cnn_2s(x_bands_2_pad_2s))
            x_bands_3_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)      
            HL = HL - self.round_layer(self.P3_cnn_2s(x_bands_3_pad_2s))
            x_bands_4_pad_2s= group_bands(LL,LL,LH,HL,HH,s_h,s_w)   
            LH = LH - self.round_layer(self.P2_cnn_2s(x_bands_4_pad_2s))
            x_bands_5_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)      
            HH = HH - self.round_layer(self.P1_cnn_2s(x_bands_5_pad_2s))
             
        x_bands_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        LH = LH - self.round_layer(self.U3_cnn_1s(x_bands_pad))
        x_bands_1_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        HL = HL - self.round_layer(self.U2_cnn_1s(x_bands_1_pad))
        x_bands_2_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        LL = LL - self.round_layer(self.U1_cnn_1s(x_bands_2_pad))
        x_bands_3_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        HL = HL - self.round_layer(self.P3_cnn_1s(x_bands_3_pad))
        x_bands_4_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        LH = LH - self.round_layer(self.P2_cnn_1s(x_bands_4_pad))
        x_bands_5_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        HH = HH - self.round_layer(self.P1_cnn_1s(x_bands_5_pad))
        
        x_bands_6 = torch.zeros(LL.shape[0],LL.shape[1],s_h,s_w,dtype=LL.dtype,device=LL.device)
        x_bands_6[:,:,0:s_h:2,0:s_w:2] = LL
        x_bands_6[:,:,1:s_h:2,0:s_w:2] = LH 
        x_bands_6[:,:,0:s_h:2,1:s_w:2] = HL 
        x_bands_6[:,:,1:s_h:2,1:s_w:2] = HH
        
        return x_bands_6

class Non_Separable_2D_WT_nonlinear_parallel(Non_Separable_2D_WT):
    def __init__(self,alpha=0.1,trainAlpha=True,lifting_struct="2D-NSWT-LCLS",isTrain=True,use_bias=False,use_init=True,wavelet="53",non_linear_arch="Proposed"):
        super(Non_Separable_2D_WT_nonlinear_parallel, self).__init__(lifting_struct=lifting_struct,isTrain=isTrain,use_bias=use_bias,use_init=use_init,wavelet=wavelet)
        
        mask_P1, mask_P2,mask_P3,mask_U1,mask_U2,mask_U3=get_lifting_operators_mask(wavelet=wavelet,stage=1)
        
        self.non_linear_arch=non_linear_arch
        
        if self.non_linear_arch == "Proposed": ccn_non_linear_arch=LCLS_Parallel
        elif self.non_linear_arch == "Variant_C": ccn_non_linear_arch=Variant_C

        self.P1_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha,mask=mask_P1)
        self.P2_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha,mask=mask_P2)
        self.P3_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha,mask=mask_P3)
        self.U3_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha,mask=mask_U3)
        self.U2_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha,mask=mask_U2)
        self.U1_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha,mask=mask_U1)
        
        if wavelet == "97":
            self.P1_cnn_nonlinear_2s, self.P2_cnn_nonlinear_2s, self.P3_cnn_nonlinear_2s,\
            self.U1_cnn_nonlinear_2s, self.U2_cnn_nonlinear_2s, self.U3_cnn_nonlinear_2s = init_linear_cnn(lifting_struct=lifting_struct,wavelet=wavelet,stage=2,use_bias=use_bias,use_init=use_init)
        
    def forward_trans(self, x):
    
        s_h = x.shape[2]
        s_w = x.shape[3]
        
        x_Img_pad=torch.cat((torch.cat((x[:,:,:,3:4],x),3),x[:,:,:,-4:-3]),3)
        x_Img_pad=torch.cat((torch.cat((x_Img_pad[:,:,3:4,:],x_Img_pad),2),x_Img_pad[:,:,-4:-3,:]),2)
        
        #batch,channel,row,collumn
        LL = x[:,:,0:s_h:2,0:s_w:2] # (2m   , 2n  )
        LH = x[:,:,1:s_h:2,0:s_w:2] # (2m+1 , 2n  )
        HL = x[:,:,0:s_h:2,1:s_w:2] # (2m   , 2n+1)
        HH = x[:,:,1:s_h:2,1:s_w:2] # (2m+1 , 2n+1)
        
        HH = HH + self.round_layer(self.P1_cnn_nonlinear_1s(x_Img_pad)+self.P1_cnn_1s(x_Img_pad))
        
        if self.lifting_struct == "2D-NSWT-LCLS":
            LH = LH + self.round_layer(self.P2_cnn_nonlinear_1s(x_Img_pad)+self.P2_cnn_1s(x_Img_pad))
            HL = HL + self.round_layer(self.P3_cnn_nonlinear_1s(x_Img_pad)+self.P3_cnn_1s(x_Img_pad))
            
        elif self.lifting_struct == "Generic":
            x_p1_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LH = LH + self.round_layer(self.P2_cnn_nonlinear_1s(x_p1_pad)+self.P2_cnn_1s(x_p1_pad))
            x_p2_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HL = HL + self.round_layer(self.P3_cnn_nonlinear_1s(x_p2_pad)+self.P3_cnn_1s(x_p2_pad))
        
        x_predicted_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)

        LL = LL + self.round_layer(self.U1_cnn_nonlinear_1s(x_predicted_pad)+self.U1_cnn_1s(x_predicted_pad))
        
        if self.lifting_struct == "2D-NSWT-LCLS":
            HL = HL + self.round_layer(self.U2_cnn_nonlinear_1s(x_predicted_pad)+self.U2_cnn_1s(x_predicted_pad))
            LH = LH + self.round_layer(self.U3_cnn_nonlinear_1s(x_predicted_pad)+self.U3_cnn_1s(x_predicted_pad))
            
        elif self.lifting_struct == "Generic":
            x_u1_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HL = HL + self.round_layer(self.U2_cnn_nonlinear_1s(x_u1_pad)+self.U2_cnn_1s(x_u1_pad))
            x_u2_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LH = LH + self.round_layer(self.U3_cnn_nonlinear_1s(x_u2_pad)+self.U3_cnn_1s(x_u2_pad))
        
        if self.wavelet == "97":
            
            x_updated_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HH = HH + self.round_layer(self.P1_cnn_nonlinear_2s(x_updated_pad)+self.P1_cnn_2s(x_updated_pad))
            
            if self.lifting_struct == "2D-NSWT-LCLS":
                LH = LH + self.round_layer(self.P2_cnn_nonlinear_2s(x_updated_pad)+self.P2_cnn_2s(x_updated_pad))
                HL = HL + self.round_layer(self.P3_cnn_nonlinear_2s(x_updated_pad)+self.P3_cnn_2s(x_updated_pad))
                
            elif self.lifting_struct == "Generic":
                x_p1_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                LH = LH + self.round_layer(self.P2_cnn_nonlinear_2s(x_p1_pad_2s)+self.P2_cnn_2s(x_p1_pad_2s))
                x_p2_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                HL = HL + self.round_layer(self.P3_cnn_nonlinear_2s(x_p2_pad_2s)+self.P3_cnn_2s(x_p2_pad_2s))
            
            x_predicted_pad_2 = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LL = LL + self.round_layer(self.U1_cnn_nonlinear_2s(x_predicted_pad_2)+self.U1_cnn_2s(x_predicted_pad_2))
            
            if self.lifting_struct == "2D-NSWT-LCLS":
                HL = HL + self.round_layer(self.U2_cnn_nonlinear_2s(x_predicted_pad_2)+self.U2_cnn_2s(x_predicted_pad_2))
                LH = LH + self.round_layer(self.U3_cnn_nonlinear_2s(x_predicted_pad_2)+self.U3_cnn_2s(x_predicted_pad_2))
            elif self.lifting_struct == "Generic":
                x_u1_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                HL = HL + self.round_layer(self.U2_cnn_nonlinear_2s(x_u1_pad_2s)+self.U2_cnn_2s(x_u1_pad_2s))
                x_u2_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                LH = LH + self.round_layer(self.U3_cnn_nonlinear_2s(x_u2_pad_2s)+self.U3_cnn_2s(x_u2_pad_2s))
        
        return LL, LH, HL, HH
    
    def inverse_trans(self, LL, LH, HL,HH):
        
        s_h = LL.shape[2]*2
        s_w = LL.shape[3]*2
        
        if self.wavelet == "97":
            x_bands_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            LH = LH - self.round_layer(self.U3_cnn_nonlinear_2s(x_bands_pad_2s)+self.U3_cnn_2s(x_bands_pad_2s))
            x_bands_1_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            HL = HL - self.round_layer(self.U2_cnn_nonlinear_2s(x_bands_1_pad_2s)+self.U2_cnn_2s(x_bands_1_pad_2s))
            x_bands_2_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            LL = LL - self.round_layer(self.U1_cnn_nonlinear_2s(x_bands_2_pad_2s)+self.U1_cnn_2s(x_bands_2_pad_2s))
            x_bands_3_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)      
            HL = HL - self.round_layer(self.P3_cnn_nonlinear_2s(x_bands_3_pad_2s)+self.P3_cnn_2s(x_bands_3_pad_2s))
            x_bands_4_pad_2s= group_bands(LL,LL,LH,HL,HH,s_h,s_w)   
            LH = LH - self.round_layer(self.P2_cnn_nonlinear_2s(x_bands_4_pad_2s)+self.P2_cnn_2s(x_bands_4_pad_2s))
            x_bands_5_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)      
            HH = HH - self.round_layer(self.P1_cnn_nonlinear_2s(x_bands_5_pad_2s)+self.P1_cnn_2s(x_bands_5_pad_2s))
             
        x_bands_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        LH = LH - self.round_layer(self.U3_cnn_nonlinear_1s(x_bands_pad)+self.U3_cnn_1s(x_bands_pad))
        
        x_bands_1_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        HL = HL - self.round_layer(self.U2_cnn_nonlinear_1s(x_bands_1_pad)+self.U2_cnn_1s(x_bands_1_pad))
        
        x_bands_2_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        LL = LL - self.round_layer(self.U1_cnn_nonlinear_1s(x_bands_2_pad)+self.U1_cnn_1s(x_bands_2_pad))
        
        x_bands_3_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        HL = HL - self.round_layer(self.P3_cnn_nonlinear_1s(x_bands_3_pad)+self.P3_cnn_1s(x_bands_3_pad))
        
        x_bands_4_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w) 
        LH = LH - self.round_layer(self.P2_cnn_nonlinear_1s(x_bands_4_pad)+self.P2_cnn_1s(x_bands_4_pad))
        
        x_bands_5_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        HH = HH - self.round_layer(self.P1_cnn_nonlinear_1s(x_bands_5_pad)+self.P1_cnn_1s(x_bands_5_pad))
        
        x_bands_6 = group_bands(LL,LL,LH,HL,HH,s_h,s_w,padding=False)  
 
        return x_bands_6

class Non_Separable_2D_WT_nonlinear_serial(Non_Separable_2D_WT):
    def __init__(self,alpha=0.1,trainAlpha=True,lifting_struct="2D-NSWT-LCLS",isTrain=True,use_bias=False,use_init=True,wavelet="53",non_linear_arch="Variant_A"):
        super(Non_Separable_2D_WT_nonlinear_serial, self).__init__(lifting_struct=lifting_struct,isTrain=isTrain,use_bias=use_bias,use_init=use_init,wavelet=wavelet)
        
        self.non_linear_arch=non_linear_arch
        
        if self.non_linear_arch == "Variant_A": ccn_non_linear_arch=Variant_A
        elif self.non_linear_arch == "Variant_B": ccn_non_linear_arch=Variant_B

        self.P1_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha)
        self.P2_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha)
        self.P3_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha)
        self.U1_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha)
        self.U2_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha)
        self.U3_cnn_nonlinear_1s = ccn_non_linear_arch(alpha=alpha,trainAlpha=trainAlpha)
        
        if wavelet == "97":
            self.P1_cnn_nonlinear_2s, self.P2_cnn_nonlinear_2s, self.P3_cnn_nonlinear_2s,\
            self.U1_cnn_nonlinear_2s, self.U2_cnn_nonlinear_2s, self.U3_cnn_nonlinear_2s = init_linear_cnn(lifting_struct=lifting_struct,wavelet=wavelet,stage=2,use_bias=use_bias,use_init=use_init)
        
    def forward_trans(self, x):
    
        s_h = x.shape[2]
        s_w = x.shape[3]
        
        x_Img_pad=torch.cat((torch.cat((x[:,:,:,3:4],x),3),x[:,:,:,-4:-3]),3)
        x_Img_pad=torch.cat((torch.cat((x_Img_pad[:,:,3:4,:],x_Img_pad),2),x_Img_pad[:,:,-4:-3,:]),2)
        
        #batch,channel,row,collumn
        LL = x[:,:,0:s_h:2,0:s_w:2] # (2m   , 2n  )
        LH = x[:,:,1:s_h:2,0:s_w:2] # (2m+1 , 2n  )
        HL = x[:,:,0:s_h:2,1:s_w:2] # (2m   , 2n+1)
        HH = x[:,:,1:s_h:2,1:s_w:2] # (2m+1 , 2n+1)
        
        HH = HH + self.round_layer(self.P1_cnn_nonlinear_1s(self.P1_cnn_1s(x_Img_pad)))
        
        if self.lifting_struct == "2D-NSWT-LCLS":
            LH = LH + self.round_layer(self.P2_cnn_nonlinear_1s(self.P2_cnn_1s(x_Img_pad)))
            HL = HL + self.round_layer(self.P3_cnn_nonlinear_1s(self.P3_cnn_1s(x_Img_pad)))
            
        elif self.lifting_struct == "Generic":
            x_p1_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LH = LH + self.round_layer(self.P2_cnn_nonlinear_1s(self.P2_cnn_1s(x_p1_pad)))
            x_p2_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HL = HL + self.round_layer(self.P3_cnn_nonlinear_1s(self.P3_cnn_1s(x_p2_pad)))
        
        x_predicted_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
        
        LL = LL + self.round_layer(self.U1_cnn_nonlinear_1s(self.U1_cnn_1s(x_predicted_pad)))
        
        if self.lifting_struct == "2D-NSWT-LCLS":
            HL = HL + self.round_layer(self.U2_cnn_nonlinear_1s(self.U2_cnn_1s(x_predicted_pad)))
            LH = LH + self.round_layer(self.U3_cnn_nonlinear_1s(self.U3_cnn_1s(x_predicted_pad)))
            
        elif self.lifting_struct == "Generic":
            x_u1_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HL = HL + self.round_layer(self.U2_cnn_nonlinear_1s(self.U2_cnn_1s(x_u1_pad)))
            x_u2_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LH = LH + self.round_layer(self.U3_cnn_nonlinear_1s(self.U3_cnn_1s(x_u2_pad)))
        
        if self.wavelet == "97":
            
            x_updated_pad = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            HH = HH + self.round_layer(self.P1_cnn_nonlinear_2s(self.P1_cnn_2s(x_updated_pad)))
            
            if self.lifting_struct == "2D-NSWT-LCLS":
                LH = LH + self.round_layer(self.P2_cnn_nonlinear_2s(self.P2_cnn_2s(x_updated_pad)))
                HL = HL + self.round_layer(self.P3_cnn_nonlinear_2s(self.P3_cnn_2s(x_updated_pad)))
                
            elif self.lifting_struct == "Generic":
                x_p1_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                LH = LH + self.round_layer(self.P2_cnn_nonlinear_2s(self.P2_cnn_2s(x_p1_pad_2s)))
                x_p2_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                HL = HL + self.round_layer(self.P3_cnn_nonlinear_2s(self.P3_cnn_2s(x_p2_pad_2s)))
            
            x_predicted_pad_2 = group_bands(x,LL,LH,HL,HH,s_h,s_w)
            LL = LL + self.round_layer(self.U1_cnn_nonlinear_2s(self.U1_cnn_2s(x_predicted_pad_2)))
            
            if self.lifting_struct == "2D-NSWT-LCLS":
                HL = HL + self.round_layer(self.U2_cnn_nonlinear_2s(self.U2_cnn_2s(x_predicted_pad_2)))
                LH = LH + self.round_layer(self.U3_cnn_nonlinear_2s(self.U3_cnn_2s(x_predicted_pad_2)))
            elif self.lifting_struct == "Generic":
                x_u1_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                HL = HL + self.round_layer(self.U2_cnn_nonlinear_2s(self.U2_cnn_2s(x_u1_pad_2s)))
                x_u2_pad_2s = group_bands(x,LL,LH,HL,HH,s_h,s_w)
                LH = LH + self.round_layer(self.U3_cnn_nonlinear_2s(self.U3_cnn_2s(x_u2_pad_2s)))
        
        return LL, LH, HL, HH
    
    def inverse_trans(self, LL, LH, HL,HH):
        
        s_h = LL.shape[2]*2
        s_w = LL.shape[3]*2
        
        if self.wavelet == "97":
            x_bands_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            LH = LH - self.round_layer(self.U3_cnn_nonlinear_2s(self.U3_cnn_2s(x_bands_pad_2s)))
            x_bands_1_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            HL = HL - self.round_layer(self.U2_cnn_nonlinear_2s(self.U2_cnn_2s(x_bands_1_pad_2s)))
            x_bands_2_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
            LL = LL - self.round_layer(self.U1_cnn_nonlinear_2s(self.U1_cnn_2s(x_bands_2_pad_2s)))
            x_bands_3_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)      
            HL = HL - self.round_layer(self.P3_cnn_nonlinear_2s(self.P3_cnn_2s(x_bands_3_pad_2s)))
            x_bands_4_pad_2s= group_bands(LL,LL,LH,HL,HH,s_h,s_w)   
            LH = LH - self.round_layer(self.P2_cnn_nonlinear_2s(self.P2_cnn_2s(x_bands_4_pad_2s)))
            x_bands_5_pad_2s = group_bands(LL,LL,LH,HL,HH,s_h,s_w)      
            HH = HH - self.round_layer(self.P1_cnn_nonlinear_2s(self.P1_cnn_2s(x_bands_5_pad_2s)))
             
        x_bands_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        LH = LH - self.round_layer(self.U3_cnn_nonlinear_1s(self.U3_cnn_1s(x_bands_pad)))
        x_bands_1_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        HL = HL - self.round_layer(self.U2_cnn_nonlinear_1s(self.U2_cnn_1s(x_bands_1_pad)))
        x_bands_2_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)
        LL = LL - self.round_layer(self.U1_cnn_nonlinear_1s(self.U1_cnn_1s(x_bands_2_pad)))
        x_bands_3_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        HL = HL - self.round_layer(self.P3_cnn_nonlinear_1s(self.P3_cnn_1s(x_bands_3_pad)))
        x_bands_4_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        LH = LH - self.round_layer(self.P2_cnn_nonlinear_1s(self.P2_cnn_1s(x_bands_4_pad)))
        x_bands_5_pad = group_bands(LL,LL,LH,HL,HH,s_h,s_w)  
        HH = HH - self.round_layer(self.P1_cnn_nonlinear_1s(self.P1_cnn_1s(x_bands_5_pad)))
        
        x_bands_6 = group_bands(LL,LL,LH,HL,HH,s_h,s_w,padding=False)  
                
        return x_bands_6

class Wavelet_Transform_2D(torch.nn.Module):
    def __init__(self,alpha=0.1,trainAlpha=True,lifting_struct="2D-NSWT-LCLS",isTrain=True,use_bias=False,use_init=True,wavelet="53",non_linear_arch="Proposed"):
        super(Wavelet_Transform_2D, self).__init__()

        if non_linear_arch=="Variant_A" or non_linear_arch=="Variant_B":
            self.lifting_operator = Non_Separable_2D_WT_nonlinear_serial
        elif non_linear_arch=="Proposed" or non_linear_arch=="Variant_C":
            self.lifting_operator = Non_Separable_2D_WT_nonlinear_parallel
            
        self.lifting = self.lifting_operator(alpha=alpha,trainAlpha=trainAlpha,lifting_struct=lifting_struct,isTrain=isTrain,use_bias=use_bias,use_init=use_init,wavelet=wavelet,non_linear_arch=non_linear_arch)
    
    def forward_trans(self, x):              
        LL, LH, HL,HH = self.lifting.forward_trans(x)
        return LL, LH, HL,HH
    
    def inverse_trans(self, LL, LH, HL,HH):
        x = self.lifting.inverse_trans(LL, LH, HL,HH)
        return x 

