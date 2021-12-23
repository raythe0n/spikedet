from torch import nn
import torch
import torch.nn.functional as F
import math

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class CausalNet(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=4,
                 blocks=2,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=64,
                 classes=1,
                 output_length=32,
                 kernel_size=3,
                 dtype=torch.FloatTensor,
                 bias=False,
                 fast=False):

        super(CausalNet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.fast = fast

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)



        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated convolutions

                self.filter_convs.append(
                    nn.Sequential(

                    #CausalConv1d(in_channels=residual_channels,
                    nn.BatchNorm1d(residual_channels),
                    nn.Conv1d(in_channels=residual_channels,
                              out_channels=dilation_channels,
                              kernel_size=kernel_size,
                              bias=bias,
                              dilation=new_dilation,
                              padding=(kernel_size-1)*new_dilation//2),

                    nn.GELU(),
                    nn.Dropout(0.5),
                ))

                #self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                #                                   out_channels=dilation_channels,
                #                                   kernel_size=kernel_size,
                #                                   bias=bias,
                #                                   dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias,
                                                 dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= kernel_size
                init_dilation = new_dilation
                new_dilation *= kernel_size



        self.head = nn.Sequential(
                        nn.Conv1d(in_channels=skip_channels,
                                  out_channels=skip_channels,
                                  kernel_size=1,
                                  bias=True),
                        nn.GELU(),
                        nn.Dropout(0.5),

                        nn.BatchNorm1d(skip_channels),
                        nn.Conv1d(in_channels=skip_channels,
                                  out_channels=classes,
                                  kernel_size=1,
                                  bias=True),
                       )

        # self.output_length = 2 ** (layers - 1)
        self.output_size = output_length
        self.receptive_field = receptive_field
        self.input_size = receptive_field + output_length - 1

    def forward(self, input, mode="normal"):
        if mode == "save":
            self.inputs = [None]* (self.blocks * self.layers)

        input = input.transpose(1, 2)

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):


            (dilation, init_dilation) = self.dilations[i]

            if mode == "save":
                self.inputs[i] = x[:,:,-(dilation*(self.kernel_size-1) + 1):]
            elif mode == "step":
                self.inputs[i] = torch.cat([self.inputs[i][:,:,1:], x], dim=2)
                x = self.inputs[i]

            # dilated convolution
            residual = x

            x = self.filter_convs[i](x)

            # parametrized skip connection
            s = self.skip_convs[i](x)
            if skip is not 0:
                skip = skip[:, :, -s.size(2):]
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual
            #x = x + residual[:, :, dilation * (self.kernel_size - 1):]

        #x = torch.relu(skip)
        #x = torch.relu(self.end_conv_1(x))
        #x = self.end_conv_2(x)

        x = self.head(skip)
        x = torch.squeeze(x, dim=-2)

        return x




class TreeNet(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=4,
                 blocks=2,
                 dilation_channels=40,
                 residual_channels=32,
                 skip_channels=64,
                 classes=1,
                 output_length=32,
                 kernel_size=3,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(TreeNet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype


        # build model
        receptive_field = 1
        init_dilation = 1

        dilation_factor = 1#2/3 #1


        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        #self.start_conv = nn.Conv1d(in_channels=self.classes,
        #                            out_channels=residual_channels,
        #                            kernel_size=1,
        #                            bias=bias)

        self.start_conv = nn.Sequential(
            #nn.BatchNorm1d(self.classes),
            nn.Conv1d(in_channels=self.classes,
                      out_channels=residual_channels,
                      kernel_size=1,
                      #bias=True),
                      bias=bias),
            #nn.Dropout(0.5),
            #nn.GELU(),
        )

        for b in range(blocks):
            additional_scope = kernel_size - 1
            real_dilation = init_dilation
            for i in range(layers):


                # dilated convolutions
                dilation = round(real_dilation)


                self.filter_convs.append(
                    nn.Sequential(

                    #CausalConv1d(in_channels=residual_channels,
                    #nn.Dropout(0.5),
                    nn.BatchNorm1d(residual_channels),
                    nn.Conv1d(in_channels=residual_channels,
                              out_channels=dilation_channels,
                              kernel_size=kernel_size,
                              bias=bias,
                              dilation=dilation,
                              padding=(kernel_size-1)*dilation//2),

                    nn.GELU(),
                    nn.Dropout(0.5),
                ))


                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Sequential(
                        #nn.BatchNorm1d(dilation_channels),
                        nn.Conv1d(in_channels=dilation_channels,
                                  out_channels=residual_channels,
                                  kernel_size=1,
                                  #bias=bias),
                                  bias=True),


                        #nn.Dropout(0.25),
                        nn.GELU(),
                        #nn.BatchNorm1d(residual_channels),
                    ))

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Sequential(
                        #nn.BatchNorm1d(dilation_channels),
                        nn.Conv1d(in_channels=dilation_channels,
                                  out_channels=skip_channels,
                                  kernel_size=1,
                                  #bias=bias),
                                  bias=True),

                        #nn.Dropout(0.25),
                        nn.GELU(),
                        #nn.BatchNorm1d(skip_channels),
                    )
                )

                receptive_field += additional_scope
                additional_scope *= dilation_factor * kernel_size

                real_dilation *= dilation_factor * kernel_size



        self.head = nn.Sequential(
                        nn.GELU(),
                        #nn.BatchNorm1d(skip_channels),
                        nn.Dropout(0.5),
                        nn.Conv1d(in_channels=skip_channels,
                                  out_channels=skip_channels,
                                  kernel_size=1,
                                  bias=True),
                        nn.GELU(),
                        nn.Dropout(0.5),

                        nn.BatchNorm1d(skip_channels),
                        nn.Conv1d(in_channels=skip_channels,
                                  out_channels=classes,
                                  kernel_size=1,
                                  bias=True),
                       )


        # self.output_length = 2 ** (layers - 1)
        self.output_size = output_length
        self.receptive_field = receptive_field
        self.input_size = receptive_field + output_length - 1

    def forward(self, input, mode="normal"):
        if mode == "save":
            self.inputs = [None]* (self.blocks * self.layers)

        input = input.transpose(1, 2)

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            # dilated convolution
            residual = x

            x = self.filter_convs[i](x)

            # parametrized skip connection
            s = self.skip_convs[i](x)
            if skip is not 0:
                skip = skip[:, :, -s.size(2):]
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual
            # x = x + residual[:, :, dilation * (self.kernel_size - 1):]

        x = self.head(skip)
        x = torch.squeeze(x, dim=-2)

        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class AttnNet(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=4,
                 blocks=2,
                 dilation_channels=40,
                 residual_channels=32,
                 skip_channels=56,
                 classes=1,
                 output_length=32,
                 kernel_size=3,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(AttnNet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype




        # build model
        receptive_field = 1
        init_dilation = 1

        dilation_factor = 1#2/3 #1


        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        #self.start_conv = nn.Conv1d(in_channels=self.classes,
        #                            out_channels=residual_channels,
        #                            kernel_size=1,
        #                            bias=bias)

        self.start_conv = nn.Sequential(
            #nn.BatchNorm1d(self.classes),
            nn.Conv1d(in_channels=self.classes,
                      out_channels=residual_channels,
                      kernel_size=1,
                      bias=True),
                      #bias=bias),
            #nn.Dropout(0.5),
            #nn.GELU(),
        )

        for b in range(blocks):
            additional_scope = kernel_size - 1
            real_dilation = init_dilation
            for i in range(layers):


                # dilated convolutions
                dilation = round(real_dilation)


                self.filter_convs.append(
                    nn.Sequential(

                    #CausalConv1d(in_channels=residual_channels,
                    #nn.Dropout(0.5),
                    nn.BatchNorm1d(residual_channels),
                    nn.Conv1d(in_channels=residual_channels,
                              out_channels=dilation_channels,
                              kernel_size=kernel_size,
                              bias=bias,
                              dilation=dilation,
                              padding=(kernel_size-1)*dilation//2),

                    #nn.BatchNorm1d(dilation_channels),

                    nn.GELU(),
                    nn.Dropout(0.5),

                    SEModule(dilation_channels, 4),
                    #nn.Dropout(0.25),
                ))

                #self.attns.append(nn.MultiheadAttention(residual_channels, num_heads=8, batch_first=True, dropout=0.5))
                #self.se.append(SEModule(dilation_channels))

                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Sequential(
                        #nn.BatchNorm1d(dilation_channels),
                        nn.Conv1d(in_channels=dilation_channels,
                                  out_channels=residual_channels,
                                  kernel_size=1,
                                  #bias=bias),
                                  bias=True),

                        #nn.BatchNorm1d(residual_channels),
                        #nn.Dropout(0.25),
                        nn.GELU(),
                        #nn.BatchNorm1d(residual_channels),
                    ))

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Sequential(
                        #nn.BatchNorm1d(dilation_channels),
                        nn.Conv1d(in_channels=dilation_channels,
                                  out_channels=skip_channels,
                                  kernel_size=1,
                                  #bias=bias),
                                  bias=True),

                        #nn.BatchNorm1d(skip_channels),
                        #nn.Dropout(0.25),
                        nn.GELU(),

                        #nn.BatchNorm1d(skip_channels),
                    )
                )

                receptive_field += additional_scope
                additional_scope *= dilation_factor * kernel_size

                real_dilation *= dilation_factor * kernel_size



        self.head = nn.Sequential(
                        nn.GELU(),
                        #SEModule(skip_channels, 4),
                        #nn.BatchNorm1d(skip_channels),
                        nn.Dropout(0.5),
                        nn.Conv1d(in_channels=skip_channels,
                                  out_channels=skip_channels//2,
                                  kernel_size=1,
                                  bias=True),
                        nn.GELU(),
                        nn.Dropout(0.5),

                        #nn.Conv1d(in_channels=skip_channels//2,
                        #          out_channels=skip_channels//4,
                        #          kernel_size=7,
                        #          padding=(7 - 1) // 2,
                                  # bias=bias),
                        #          bias=True),

                        nn.BatchNorm1d(skip_channels//2),
                        nn.Conv1d(in_channels=skip_channels//2,
                                  out_channels=classes,
                                  kernel_size=1,
                                  bias=True),
                       )

        self.linear = nn.Linear(skip_channels//2 * 32, 32)


        # self.output_length = 2 ** (layers - 1)
        self.output_size = output_length
        self.receptive_field = receptive_field
        self.input_size = receptive_field + output_length - 1

    def forward(self, input, mode="normal"):
        if mode == "save":
            self.inputs = [None]* (self.blocks * self.layers)

        input = input.transpose(1, 2)

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            # dilated convolution
            residual = x

            x = self.filter_convs[i](x)


            # parametrized skip connection
            s = self.skip_convs[i](x)
            if skip is not 0:
                skip = skip[:, :, -s.size(2):]
            skip = s + skip

            x = self.residual_convs[i](x)

            #qkv = x.transpose(-2, -1)
            #attn, att_weights = self.attns[i](qkv, qkv, qkv)
            #x = attn.transpose(-2, -1)

            x = x + residual
            # x = x + residual[:, :, dilation * (self.kernel_size - 1):]

        x = self.head(skip)
        #x = self.linear(x.flatten(-2))
        x = torch.squeeze(x, dim=-2)

        return x


class Residual(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, skip_channels, dilation, kernel_size, padding=0):
        super(Residual, self).__init__()
        self.padding = padding

        if out_channels > 0:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim,
                          out_channels=out_channels,
                          kernel_size=1,
                          # bias=bias),
                          bias=True),

                # nn.BatchNorm1d(residual_channels),
                nn.GELU(),
            )
        else:
            self.residual = None

        self.skip = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim,
                      out_channels=skip_channels,
                      kernel_size=1,
                      # bias=bias),
                      bias=True),

            # nn.BatchNorm1d(residual_channels),
            nn.GELU(),
        )

        layers = []
        #if in_channels != hidden_dim:
        #    layers += [ # pw
        #        nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, bias=False ),
        #    ]

        layers += [
            # pw
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, bias=False),
            # dw
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, (kernel_size - 1) // 2, dilation=dilation,
                      padding=(kernel_size - 1) * dilation // 2, groups=hidden_dim, bias=False),

            nn.GELU(),
            nn.Dropout(0.5),
            # Squeeze-and-Excite
            SEModule(hidden_dim),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)

        z = x + self.residual(y) if self.residual is not None else x
        #Remove
        y = y[..., self.padding:-self.padding]
        s = self.skip(y)
        return (z, s)

def lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (False), or if it is the last value (True).
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, False
        last = val
    # Report the last value.
    yield last, True

'''
SpikeNet Model
'''

class SpikeNet(nn.Module):

    def __init__(self,
                 layers=4,
                 blocks=2,
                 dilation_channels=40,
                 residual_channels=32,
                 skip_channels=64,
                 classes=1,
                 kernel_size=3,
                 padding=4,
                 dilation_factor=1.0,#2/3 #1
                 bias=False):

        super(SpikeNet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.padding = padding


        # build model
        receptive_field = 1
        init_dilation = 1

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.filters = nn.ModuleList()

        in_channels = 1
        out_channels = residual_channels

        for b, last_block in lookahead(range(blocks)):
            additional_scope = kernel_size - 1
            real_dilation = init_dilation
            for i, last_layer in lookahead(range(layers)):

                # dilated convolutions
                dilation = round(real_dilation)

                if last_block and last_layer:
                    out_channels = 0

                self.filters.append(
                    Residual(in_channels,
                             dilation_channels,
                             out_channels,
                             skip_channels,
                             dilation,
                             kernel_size, padding=padding)
                )

                in_channels = out_channels

                receptive_field += additional_scope
                additional_scope *= dilation_factor * kernel_size

                real_dilation *= dilation_factor * kernel_size


        self.head = nn.Sequential(
                        nn.GELU(),
                        #nn.BatchNorm1d(skip_channels),
                        nn.Dropout(0.5),
                        nn.Conv1d(in_channels=skip_channels,
                                  out_channels=skip_channels,
                                  kernel_size=1,
                                  bias=True),

                        nn.GELU(),
                        nn.Dropout(0.5),

                        nn.BatchNorm1d(skip_channels),
                        nn.Conv1d(in_channels=skip_channels,
                                  out_channels=classes,
                                  kernel_size=1,
                                  bias=True),
                       )


        self.receptive_field = receptive_field
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                #n = m.kernel_size[0] * m.out_channels
                #w = math.sqrt(2. / n)
                #nn.init.uniform_(m.weight, -w, w)
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(3))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input):

        x = input.unsqueeze(-2)
        #x = input.transpose(1, 2)

        skip = 0

        for i in range(self.blocks * self.layers):

            x, s = self.filters[i](x)
            skip += s

        x = self.head(skip)

        x = torch.squeeze(x, dim=-2)

        #if not self.training:
        #    x = torch.sigmoid_(x)

        return x