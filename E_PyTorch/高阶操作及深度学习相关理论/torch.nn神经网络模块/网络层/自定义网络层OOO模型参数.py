import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.nn.init as init


class MyLinear(nn.Module):
    """参考自:torch.nn.Module.Linear"""

    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # A kind of Tensor that is to be considered a module parameter.
        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features),
                        requires_grad=True,  # if the parameter requires gradient.
                        **factory_kwargs))  # 网络层学习参数
        if bias:
            self.bias = nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=True)
        else:
            # "Adds a parameter to the module
            self.register_parameter(name='bias', param=None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
