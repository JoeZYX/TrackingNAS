import os
import torch
import torch.nn as nn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)



def qat_init_model_manu(model, num_bits, calib_method):
    # print(model)



    
    conv2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)

    convtrans2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    convtrans2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)

    for k, m in model.named_modules():
        # print(k)
        # print(k, m)
        if isinstance(m, nn.Conv2d):
            # print("in_channel = {}".format(m.in_channels))
            # print("out_channel = {}".format(m.out_channels))
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              quant_desc_input = conv2d_input_default_desc,
                                              quant_desc_weight = conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            # print("in_channel = {}".format(m.in_channels))
            # print("out_channel = {}".format(m.out_channels))
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                       out_channels,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       quant_desc_input = convtrans2d_input_default_desc,
                                                       quant_desc_weight = convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            # print("dilation = {}".format(m.dilation))
            # print("ceil mode = {}".format(m.ceil_mode))
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input = conv2d_input_default_desc)
            set_module(model, k, quant_maxpool2d)
        else:
            # module can not be quantized, continue
            continue