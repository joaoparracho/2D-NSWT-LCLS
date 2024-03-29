import torch
from torch.nn import functional as F

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = torch.logical_or(x >= 1e-6, g < 0.0)
        t = pass_through_if+0.0

        return grad1 * t

class Distribution_for_entropy2(torch.nn.Module):
    def __init__(self):
        super(Distribution_for_entropy2, self).__init__()

    def forward(self, x, p_dec): 
        channel = p_dec.size()[1]
        if channel % 3 != 0:
            raise ValueError(
                "channel number must be multiple of 3")
        gauss_num = channel // 3
        
        # keep the weight  summation of prob == 1
        probs = p_dec[:,gauss_num*2:,:,:]
        means = p_dec[:,0:gauss_num,:,:]
        scale = p_dec[:,gauss_num:gauss_num*2,:,:]
  
        probs = F.softmax(probs, dim=1)
        scale = torch.clamp(torch.abs(scale),min=1e-6)
        
        gauss_list = []
        for i in range(gauss_num):
            gauss_list.append(torch.distributions.normal.Normal(means[:,i:i+1,:,:], scale[:,i:i+1,:,:]))

        likelihood_list = []
        for i in range(gauss_num):
            likelihood_list.append(torch.abs(gauss_list[i].cdf(x + 0.5)-gauss_list[i].cdf(x-0.5)))

        likelihoods = 0
        for i in range(gauss_num):
            likelihoods += probs[:,i:i+1,:,:] * likelihood_list[i]
            
        return likelihoods

class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskResBlock(torch.nn.Module):
    def __init__(self, internal_channel):
        super(MaskResBlock, self).__init__()

        self.conv1 = MaskedConv2d('B', in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.conv2 = MaskedConv2d('B', in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x[:,:,2:-2,2:-2]


class ResBlock(torch.nn.Module):
    def __init__(self, internal_channel):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1((x))
        out = self.relu(out)
        out = self.conv2((out))
        return out + x[:,:,2:-2,2:-2]


class PixelCNN(torch.nn.Module):
    def __init__(self,nec=128):
        super(PixelCNN, self).__init__()

        self.internal_channel = nec
        self.num_params = 9

        self.relu = torch.nn.ReLU(inplace=False)

        self.padding_constant = torch.nn.ConstantPad2d(6, 0)
        self.conv_pre = MaskedConv2d('A', in_channels=1, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.res1 = MaskResBlock(self.internal_channel)
        self.res2 = MaskResBlock(self.internal_channel)
        self.conv_post = MaskedConv2d('B', in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1,padding=0)

        def infering():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.num_params, kernel_size=1, stride=1, padding=0)
            )
        self.infer = infering()
        self.gaussin_entropy_func = Distribution_for_entropy2()

    def forward_train(self, x):
        x = x 

        lable = x

        x = self.padding_constant(x)
        x = self.conv_pre(x)
        conv1 = x
        x = self.res1(x)
        x = self.res2(x)
        x = conv1[:,:,4:-4,4:-4] + x
        x = self.conv_post(x)
        x = self.relu(x)

        params = self.infer(x)
        
        prob = self.gaussin_entropy_func(lable, params)
        prob = Low_bound.apply(prob)

        bits = -torch.sum(torch.log2(prob))

        return bits
    
    def forward_inf(self, x, range_low_bond, range_high_bond):

        label = torch.arange(start=range_low_bond, end=range_high_bond + 1, dtype=torch.float,
                             device=torch.device('cuda'))
        label = label.unsqueeze(0).unsqueeze(0).unsqueeze(0)#/255.
        x = x
        # label has the size of [1,1,1,range]

        # x = self.padding_constant(x)
        x = self.conv_pre(x)
        conv1 = x
        x = self.res1(x)
        x = self.res2(x)
        x = conv1[:,:,4:-4,4:-4] + x
        x = self.conv_post(x)
        x = self.relu(x)

        params = self.infer(x)
        # params has the size of [N, 9, 1, 1]
        params = params.repeat(1, 1, 1, range_high_bond - range_low_bond + 1)
        # params has the size of [N, 9, 1, range]
        
        prob = self.gaussin_entropy_func(label, params)
        # prob has the size of [N, 1, 1, range]

        prob = prob + 1e-5

        return prob.squeeze(1).squeeze(1)


class PixelCNN_Context(torch.nn.Module):
    def __init__(self, context_num,nec=128):
        super(PixelCNN_Context, self).__init__()

        self.internal_channel = nec
        self.num_params = 9

        self.relu = torch.nn.ReLU(inplace=False)

        self.conv_pre = MaskedConv2d('A', in_channels=1, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.res1 = MaskResBlock(self.internal_channel)
        self.res2 = MaskResBlock(self.internal_channel)
        self.conv_post = MaskedConv2d('B', in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1,padding=0)

        self.padding_reflect = torch.nn.ReflectionPad2d(6)
        self.padding_constant = torch.nn.ConstantPad2d(6, 0)
        self.conv_pre_c = torch.nn.Conv2d(in_channels=context_num, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.res1_c = ResBlock(self.internal_channel)
        self.res2_c = ResBlock(self.internal_channel)

        def infering():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.num_params, kernel_size=1, stride=1, padding=0)
            )
        self.infer = infering()
        self.gaussin_entropy_func = Distribution_for_entropy2()

    def forward_train(self, x, context):
        x = x 
        context = context

        lable = x
        x = self.padding_constant(x)
        context = self.padding_reflect(context)
        x = self.conv_pre(x)
        conv1 = x
        context = self.conv_pre_c((context))
        x = x + context

        x = self.res1(x)
        context = self.res1_c(context)
        x = x + context
        x = self.res2(x)
        context = self.res2_c(context)
        x = x + context

        x = conv1[:,:,4:-4,4:-4] + x
        x = self.conv_post(x)
        x = self.relu(x)

        params = self.infer(x)
        
        prob = self.gaussin_entropy_func(lable, params)
        prob = Low_bound.apply(prob)
        bits = -torch.sum(torch.log2(prob))

        return bits

    def forward_inf(self, x, context, range_low_bond, range_high_bond):

        label = torch.arange(start=range_low_bond, end=range_high_bond + 1, dtype=torch.float,
                             device=torch.device('cuda'))
        label = label.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x = x 
        context = context 
        # label has the size of [1,1,1,range]

        # x = self.padding_constant(x)
        # context = self.padding_reflect(context)
        x = self.conv_pre(x)
        conv1 = x
        context = self.conv_pre_c((context))
        x = x + context

        x = self.res1(x)
        context = self.res1_c(context)
        x = x + context
        x = self.res2(x)
        context = self.res2_c(context)
        x = x + context

        x = conv1[:,:,4:-4,4:-4] + x
        x = self.conv_post(x)
        x = self.relu(x)

        params = self.infer(x)
        # params has the size of [N, 9, 1, 1]
        params = params.repeat(1, 1, 1, range_high_bond - range_low_bond + 1)
        # params has the size of [N, 9, 1, range]

        prob = self.gaussin_entropy_func(label, params)
        # prob has the size of [N, 1, 1, range]

        prob = prob + 1e-5

        return prob.squeeze(1).squeeze(1)

class Resblock_tanh(torch.nn.Module):
    def __init__(self,n_channels):
        
        super(Resblock_tanh, self).__init__()
        self.padding = torch.nn.ReflectionPad2d(1)
        
        self.dynamic_range = float(2**8)
 
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=0)
        
    
    def forward(self, x):
        
        x_1 = self.conv1(self.padding(self.tanh(x/self.dynamic_range)))
        x_2 = self.tanh(x_1)
        out = self.conv2(self.padding(x_2))
        
        return out*self.dynamic_range+x
 