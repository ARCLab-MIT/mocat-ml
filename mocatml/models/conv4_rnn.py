__all__ = ['AddCoords', 'CoordConv', 'Conv4GRU_cell', 'TimeDistributed', 'Encoder4d', 'UpsampleBlock', 'conditional_crop_pad',
           'Decoder4d', 'Stack4Unstack', 'Simple4Model', 'StackLoss', 'PartialStackLoss', 'MultiImageDice', 'Conv4d']

from fastai.vision.all import *
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

# https://github.com/ZhengyuLiang24/Conv4d-PyTorch/blob/main/Conv4d.py
class Conv4d(Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple],
                 stride:[int, tuple] = (1, 1, 1, 1),
                 padding:[int, tuple] = (0, 0, 0, 0),
                 dilation:[int, tuple] = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):

        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::],
                                     bias=False)
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        # del self.weight


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape) 

        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out


class AddCoords(Module):

    def __init__(self, with_r=False):
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


@delegates(nn.Conv2d)
class CoordConv(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        self.addcoords = AddCoords(with_r=True)
        in_size = in_channels+2
        self.conv = nn.Conv2d(in_size+1, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class Conv4Layer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    @delegates(Conv4d)
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=defaults.activation, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv = Conv4d(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)


# %% ../../nbs_lib/models.conv_rnn.ipynb 7
class Conv4GRU_cell(Module):
    def __init__(self, in_ch, out_ch, ks=3, debug=False):
        self.in_ch = in_ch
        # kernel_size of input_to_state equals state_to_state
        self.ks = ks
        self.out_ch = out_ch
        self.debug = debug
        self.padding = (ks - 1) // 2
        self.conv1 = nn.Sequential(Conv4d(self.in_ch + self.out_ch,2 * self.out_ch, self.ks, 1,self.padding),
                                   nn.GroupNorm(2 * self.out_ch // 8, 2 * self.out_ch))
        self.conv2 = nn.Sequential(Conv4d(self.in_ch + self.out_ch,self.out_ch, self.ks, 1, self.padding),
                                   nn.GroupNorm(self.out_ch // 8, self.out_ch))

    def forward(self, inputs, hidden_state=None):
        "inputs shape: (bs, seq_len, ch, w, h)"
        # bs, seq_len, ch, w, h = inputs.shape
        bs, seq_len, ch, d1, d2, d3, d4 = inputs.shape
        if hidden_state is None:
            htprev = self.initHidden(bs, self.out_ch, d1, d2, d3, d4)
            if self.debug: print(f'htprev: {htprev.shape}')
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            x = inputs[:, index, ...]
            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)          
            zgate, rgate = torch.split(gates, self.out_ch, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)
            combined_2 = torch.cat((x, r * htprev),1)
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner, dim=1), htnext
    def __repr__(self): return f'ConvGRU_cell(in={self.in_ch}, out={self.out_ch}, ks={self.ks})'
    def initHidden(self, bs, ch, d1, d2, d3, d4): return one_param(self).new_zeros(bs, ch, d1, d2, d3, d4)

# %% ../../nbs_lib/models.conv_rnn.ipynb 16
class TimeDistributed(Module):
    "Applies a module over tdim identically for each step" 
    def __init__(self, module, low_mem=False, tdim=1):
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim
        
    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*args)
        else:
            # Only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]

            # Process non-None arguments only
            processed_args = [x.view(bs*seq_len, *x.shape[2:]) for x in args if x is not None]
            out = self.module(*processed_args, **kwargs)

            out_shape = out.shape
            return out.view(bs, seq_len, *out_shape[1:])
    
    def low_mem_forward(self, *args, **kwargs):                                           
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args if x is not None]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out, dim=self.tdim)
    
    def __repr__(self):
        return f'TimeDistributed({self.module})'


# %% ../../nbs_lib/models.conv_rnn.ipynb 24
class Encoder4d(Module):
    def __init__(self, n_in=1, szs=[16,64,96], ks=3, rnn_ks=5, act=nn.ReLU, norm=None, coord_conv=False, debug=False):
        self.debug = debug
        convs = []
        rnns = []
        if coord_conv: 
            self.coord_conv = TimeDistributed(CoordConv(n_in, 8, kernel_size=1))
            szs = [8]+szs
        else: 
            self.coord_conv = Lambda(noop)
            szs = [n_in]+szs
        for ni, nf in zip(szs[0:-1], szs[1:]):
            # convs.append(Conv4d(ni, nf, kernel_size=ks))
            convs.append(Conv4Layer(ni, nf, ks=ks, stride=1 if ni==szs[0] else 2, padding=ks//2, act_cls=act, norm_type=norm))
            rnns.append(Conv4GRU_cell(nf, nf, ks=rnn_ks))
        self.convs = nn.ModuleList(TimeDistributed(conv) for conv in convs)
        self.rnns = nn.ModuleList(rnns)
        

    def forward_by_stage(self, inputs, conv, rnn):
        if self.debug: 
            print(f' Layer: {rnn}')
            print(' inputs: ', inputs.shape)
        inputs = conv(inputs)
        if self.debug: print(' after_convs: ', inputs.shape)
        outputs_stage, state_stage = rnn(inputs, None)
        if self.debug: print(' output_stage: ', outputs_stage.shape)
        return outputs_stage, state_stage


    def forward(self, inputs):
        "inputs.shape bs,seq_len,1,64,64"
        hidden_states = []
        outputs = []
        inputs = self.coord_conv(inputs)
        for i, (conv, rnn) in enumerate(zip(self.convs, self.rnns)):
            if self.debug: print('stage: ',i)
            inputs, state_stage = self.forward_by_stage(inputs, conv, rnn)
            outputs.append(inputs)
            hidden_states.append(state_stage)
        return outputs, hidden_states
    

def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,d1,d2,d3,d4 = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,d1,d2,d3,d4])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,d1,d2,d3,d4]).transpose(0, 1)
    return k

class PixelShuffle_ICNR4d(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, act_cls=defaults.activation):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [Conv4Layer(ni, nf*(scale**4), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=0),
                  PixelShuffle4d(scale)]
        if norm_type == NormType.Weight:
            layers[0][0].weight_v.data.copy_(icnr_init(layers[0][0].weight_v.data))
            layers[0][0].weight_g.data.copy_(((layers[0][0].weight_v.data**2).sum(dim=[1,2,3,4])**0.5)[:,None,None,None])
        else:
            layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        super().__init__(*layers)


class PixelShuffle4d(nn.Module):
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, d1, d2, d3, d4 = input.size()
        nOut = channels // self.scale ** 4

        a1 = d1 * self.scale
        a2 = d2 * self.scale
        a3 = d3 * self.scale
        a4 = d4 * self.scale


        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, self.scale, d1, d2, d3, d4)

        output = input_view.permute(0, 1, 6, 2, 7, 3, 8, 4, 9, 5).contiguous()

        return output.view(batch_size, nOut, a1, a2, a3, a4)


# %% ../../nbs_lib/models.conv_rnn.ipynb 28
class UpsampleBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    @delegates(Conv4Layer.__init__)
    def __init__(self, in_ch, out_ch, residual=False, blur=False, act_cls=defaults.activation,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, debug=False, **kwargs):
        store_attr()
        self.shuf = PixelShuffle_ICNR4d(in_ch, in_ch//2, blur=blur, act_cls=act_cls, norm_type=norm_type)
        ni = in_ch//2 if not residual else in_ch//2 + out_ch  #the residual has out_ch (normally in_ch//2)
        nf = out_ch
        self.conv1 = Conv4Layer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = Conv4Layer(nf, nf, act_cls=act_cls, norm_type=norm_type,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)

        # self.conv1 = Conv4d(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        # self.conv2 = Conv4d(nf, nf, act_cls=act_cls, norm_type=norm_type,
        #                        xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = act_cls()

        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    def __repr__(self): return (f'UpsampleBLock(in={self.in_ch}, out={self.out_ch}, blur={self.blur}, residual={self.residual}, '
                                f'act={self.act_cls()}, attn={self.self_attention}, norm={self.norm_type})')
    
    def forward(self, up_in, side_in=None):
        up_out = self.shuf(up_in)
        if side_in is not None:
            if self.debug: print(f'up_out: {up_out.shape}, side_in: {side_in.shape}')
            assert up_out.shape[-2:] == side_in.shape[-2::], 'residual shape does not match input'
            up_out = torch.cat([up_out, self.bn(side_in)], dim=1)
        if self.debug: print(f'up_out: {up_out.shape}')
        return self.conv2(self.conv1(up_out))

# %% ../../nbs_lib/models.conv_rnn.ipynb 30
import torch
import torch.nn.functional as F

def conditional_crop_pad(tensor, target_height, target_width):
    """
    Conditionally crops or pads the input tensor to match the target height and width.
    Args:
        tensor (Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)
        target_height (int): Target height
        target_width (int): Target width
    Returns:
        Tensor: Adjusted tensor
    """
    height, width = tensor.shape[-2], tensor.shape[-1]

    # Height adjustment
    if height > target_height:
        # Crop height
        start_h = (height - target_height) // 2
        tensor = tensor[:, :, :, start_h:start_h + target_height, :]
    elif height < target_height:
        # Pad height
        padding_h = (target_height - height) // 2
        tensor = F.pad(tensor, (0, 0, 0, 0, padding_h, padding_h), "constant", 0)

    # Width adjustment
    if width > target_width:
        # Crop width
        start_w = (width - target_width) // 2
        tensor = tensor[:, :, :, :, start_w:start_w + target_width]
    elif width < target_width:
        # Pad width
        padding_w = (target_width - width) // 2
        tensor = F.pad(tensor, (0, 0, padding_w, padding_w, 0, 0), "constant", 0)

    return tensor

# In your Decoder's forward method, you would use this function like so:
class Decoder(nn.Module):
    # ... (other parts of the Decoder class)

    def forward(self, dec_input, hidden_states, enc_outs):
        # ... existing logic ...

        output = self.head(dec_input)
        output_adjusted = conditional_crop_pad(output, 36, 99)
        return output_adjusted



class Decoder4d(Module):
    def __init__(self, n_out=1, szs=[96,64,16], ks=3, rnn_ks=5, act=nn.ReLU, 
                 blur=False, attn=False, 
                 norm=None, debug=False):
        self.debug = debug
        deconvs = []
        rnns = []
        szs = szs
        for ni, nf in zip(szs[0:-1], szs[1:]):
            deconvs.append(UpsampleBlock(ni, nf, blur=blur, self_attention=attn, act_cls=act, norm_type=norm))
            rnns.append(Conv4GRU_cell(ni, ni, ks=rnn_ks))
        
        #last layer
        deconvs.append(Conv4Layer(szs[-1], szs[-1], ks, padding=ks//2, act_cls=act, norm_type=norm))
        self.deconvs = nn.ModuleList(TimeDistributed(conv) for conv in deconvs)
        self.rnns = nn.ModuleList(rnns)
        self.head = TimeDistributed(Conv4d(szs[-1], n_out, kernel_size=1))

    def forward_by_stage(self, inputs, state, deconv, rnn, side_in=None):
        if self.debug: 
            print(f' Layer: {rnn}')
            print(' inputs:, state: ', inputs.shape, state.shape)
        inputs, state_stage = rnn(inputs, state)
        if self.debug: 
            print(' after rnn: ', inputs.shape)
            print(f' Layer: {deconv}')
            print(f' before Upsample: inputs are {inputs.shape}, side_in is \
                  {side_in.shape if side_in is not None else None}')
        outputs_stage = deconv(inputs, side_in)
        if self.debug: print(' after_deconvs: ', outputs_stage.shape)
        return outputs_stage, state_stage
    
    def forward(self, dec_input, hidden_states, enc_outs):
        # Capture the target spatial dimensions from enc_outs
        target_height, target_width = enc_outs[0].shape[-2], enc_outs[0].shape[-1]
        if self.debug: print(f'target_height: {target_height}, target_width: {target_width}')

        enc_outs = [None]+enc_outs[:-1]
        for i, (state, conv, rnn, enc_out) in enumerate(zip(hidden_states[::-1], self.deconvs, self.rnns, enc_outs[::-1])):
            if self.debug: print(f'\nStage: {i} ---------------------------------')
            # dec_input, state_stage = self.forward_by_stage(dec_input, state, 
            #                                                conv, rnn, side_in=enc_out)
            dec_input, state_stage = self.forward_by_stage(dec_input, state, 
                                                           conv, rnn, side_in=None)
        output = self.head(dec_input)
        # Resize the output to the expected dimensions (padding/cropping)
        output_adjusted = conditional_crop_pad(output, target_height, target_width)
        return output_adjusted

# %% ../../nbs_lib/models.conv_rnn.ipynb 36
def _unbind_densities(x, dim=1):
    "only unstack densities"
    if isinstance(x, torch.Tensor): 
        if len(x.shape)>=4:
            return x.unbind(dim=dim)
    return x

# %% ../../nbs_lib/models.conv_rnn.ipynb 38
class Stack4Unstack(Module):
    "Stack together inputs, apply module, unstack output"
    def __init__(self, module, dim=1):
        self.dim = dim
        self.module = module
    
    @staticmethod
    def unbind_densities(x, dim=1): return _unbind_densities(x, dim)
    def forward(self, *args):
        inputs = [torch.stack(x, dim=self.dim) for x in args]
        outputs = self.module(*inputs)
        if isinstance(outputs, (tuple, list)):
            return [self.unbind_densities(output, dim=self.dim) for output in outputs]
        else: return outputs.unbind(dim=self.dim)

# %% ../../nbs_lib/models.conv_rnn.ipynb 39
class Simple4Model(Module):
    "Simple Encoder/Decoder module"
    def __init__(self, n_in=1, n_out=1, szs=[16,64,96], ks=3, rnn_ks=5, 
                 act=nn.ReLU, blur=False, attn=False, norm=None, strategy='zero', 
                 coord_conv=False, debug=False):
        self.strategy = strategy
        self.encoder = Encoder4d(n_in, szs, ks, rnn_ks, act, norm, coord_conv, debug)
        self.decoder = Decoder4d(n_out, szs[::-1], ks, rnn_ks, act, blur, attn, norm, debug)
    def forward(self, x):
        enc_outs, h = self.encoder(x)
        if self.strategy == 'zero':
            dec_in = one_param(self).new_zeros(*enc_outs[-1].shape)
        elif self.strategy == 'encoder':
            dec_in = enc_outs[-1]
        return self.decoder(dec_in, h, enc_outs)

# %% ../../nbs_lib/models.conv_rnn.ipynb 44
# def StackLoss(loss_func=MSELossFlat(), axis=-1):
#     def _inner_loss(x,y):
#         x = torch.cat(x, axis)
#         y = torch.cat(y, axis)
#         return loss_func(x,y)
#     return _inner_loss

# %% ../../nbs_lib/models.conv_rnn.ipynb 45
class StackLoss(nn.Module):
    def __init__(self, loss_func=MSELossFlat(), axis=-1):
        super().__init__()
        self.loss_func = loss_func
        self.axis = axis

    def forward(self, x, y):
        x = torch.cat(x, self.axis)
        y = torch.cat(y, self.axis)
        return self.loss_func(x, y)

# %% ../../nbs_lib/models.conv_rnn.ipynb 51
class PartialStackLoss(StackLoss):
    """StackLoss but only in a subset of the elements of the list"""
    @delegates(StackLoss.__init__)
    def __init__(self, idxs, **kwargs):
        super().__init__(**kwargs)
        self.idxs = idxs

    def forward(self, x, y, **kwargs):
        return super().forward([x[i] for i in self.idxs], 
                               [y[i] for i in self.idxs],
                               **kwargs)

# %% ../../nbs_lib/models.conv_rnn.ipynb 54
class MultiImageDice(Metric):
    "Dice coefficient metric for binary target in segmentation"
    def __init__(self, axis=1): self.axis = axis
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        x = torch.cat(learn.pred, -1)
        y = torch.cat(learn.y, -1)
#         print(type(x), type(y), x.shape, y.shape)
        pred,targ = flatten_check(x.argmax(dim=self.axis), y)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()

    @property
    def value(self): return 2. * self.inter/self.union if self.union > 0 else None
