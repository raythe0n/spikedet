from torch import nn
import torch
import torch.nn.functional as F
import math

from spikedet.core.utils import lookahead

"""
Squeeze and exite Layer
"""
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
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

"""
Residual layer
"""
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
            SELayer(hidden_dim),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)

        z = x + self.residual(y) if self.residual is not None else x
        #Remove
        y = y[..., self.padding:-self.padding]
        s = self.skip(y)
        return (z, s)


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
        skip = 0

        for i in range(self.blocks * self.layers):
            x, s = self.filters[i](x)
            skip += s

        x = self.head(skip)
        x = torch.squeeze(x, dim=-2)

        #if not self.training:
        #    x = torch.sigmoid_(x)

        return x