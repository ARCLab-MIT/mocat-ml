# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs_lib/models.conv_rnn.ipynb.

# %% auto 0
__all__ = ['AddCoords', 'CoordConv', 'ConvGRU_cell', 'TimeDistributed', 'Encoder', 'UpsampleBlock', 'conditional_crop_pad',
           'Decoder', 'StackUnstack', 'SimpleModel', 'StackLoss', 'PartialStackLoss', 'MultiImageDice']

# %% ../../nbs_lib/models.conv_rnn.ipynb 2
from fastai.vision.all import *
import torch.nn.functional as F

# %% ../../nbs_lib/models.conv_rnn.ipynb 5
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

# %% ../../nbs_lib/models.conv_rnn.ipynb 7
class ConvGRU_cell(Module):
    def __init__(self, in_ch, out_ch, ks=3, debug=False):
        self.in_ch = in_ch
        # kernel_size of input_to_state equals state_to_state
        self.ks = ks
        self.out_ch = out_ch
        self.debug = debug
        self.padding = (ks - 1) // 2
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_ch + self.out_ch,2 * self.out_ch, self.ks, 1,self.padding),
                                   nn.GroupNorm(2 * self.out_ch // 8, 2 * self.out_ch))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_ch + self.out_ch,self.out_ch, self.ks, 1, self.padding),
                                   nn.GroupNorm(self.out_ch // 8, self.out_ch))

    def forward(self, inputs, hidden_state=None):
        "inputs shape: (bs, seq_len, ch, w, h)"
        bs, seq_len, ch, w, h = inputs.shape
        if hidden_state is None:
            htprev = self.initHidden(bs, self.out_ch, w, h)
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
    def initHidden(self, bs, ch, w, h): return one_param(self).new_zeros(bs, ch, w, h)

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
class Encoder(Module):
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
            convs.append(ConvLayer(ni, nf, ks=ks, stride=1 if ni==szs[0] else 2, padding=ks//2, act_cls=act, norm_type=norm))
            rnns.append(ConvGRU_cell(nf, nf, ks=rnn_ks))
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

# %% ../../nbs_lib/models.conv_rnn.ipynb 28
class UpsampleBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    @delegates(ConvLayer.__init__)
    def __init__(self, in_ch, out_ch, residual=False, blur=False, act_cls=defaults.activation,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, debug=False, **kwargs):
        store_attr()
        self.shuf = PixelShuffle_ICNR(in_ch, in_ch//2, blur=blur, act_cls=act_cls, norm_type=norm_type)
        ni = in_ch//2 if not residual else in_ch//2 + out_ch  #the residual has out_ch (normally in_ch//2)
        nf = out_ch
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(nf, nf, act_cls=act_cls, norm_type=norm_type,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
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



class Decoder(Module):
    def __init__(self, n_out=1, szs=[96,64,16], ks=3, rnn_ks=5, act=nn.ReLU, 
                 blur=False, attn=False, 
                 norm=None, debug=False):
        self.debug = debug
        deconvs = []
        rnns = []
        szs = szs
        for ni, nf in zip(szs[0:-1], szs[1:]):
            deconvs.append(UpsampleBlock(ni, nf, blur=blur, self_attention=attn, act_cls=act, norm_type=norm))
            rnns.append(ConvGRU_cell(ni, ni, ks=rnn_ks))
        
        #last layer
        deconvs.append(ConvLayer(szs[-1], szs[-1], ks, padding=ks//2, act_cls=act, norm_type=norm))
        self.deconvs = nn.ModuleList(TimeDistributed(conv) for conv in deconvs)
        self.rnns = nn.ModuleList(rnns)
        self.head = TimeDistributed(nn.Conv2d(szs[-1], n_out, kernel_size=1))

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
class StackUnstack(Module):
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
class SimpleModel(Module):
    "Simple Encoder/Decoder module"
    def __init__(self, n_in=1, n_out=1, szs=[16,64,96], ks=3, rnn_ks=5, 
                 act=nn.ReLU, blur=False, attn=False, norm=None, strategy='zero', 
                 coord_conv=False, debug=False):
        self.strategy = strategy
        self.encoder = Encoder(n_in, szs, ks, rnn_ks, act, norm, coord_conv, debug)
        self.decoder = Decoder(n_out, szs[::-1], ks, rnn_ks, act, blur, attn, norm, debug)
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
