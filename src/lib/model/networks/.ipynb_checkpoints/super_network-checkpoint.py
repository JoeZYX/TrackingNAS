import torch
import torch.nn as nn
import numpy as np
import math
#from pytorch_quantization import nn as quant_nn
#from pytorch_quantization.tensor_quant import QuantDescriptor

# ----------------- BackBone ----------------

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# 1. basic convolution operation 
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.ReLU()  # default activation

    def __init__(self, c_in, c_out, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True, bias = True):
        
        super().__init__()
        
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, autopad(kernel_size, padding), groups=groups, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        if isinstance(act, str):
            act = eval(act)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 2. Residual convolutional block ---> 2 convolutional + skip connection

class BottleRep(nn.Module):

    def __init__(self, c_in, c_out, basic_block=Conv, quant = None, weight=False, shortcut=True, conv1_cfg = None, conv2_cfg = None):
        super().__init__()
        assert conv1_cfg is not None
        assert conv2_cfg is not None
        self.quant = quant
        print(" check ------------:", self.quant)
        self.conv1 = basic_block(c_in, c_out, **conv1_cfg)
        self.conv2 = basic_block(c_in, c_out, **conv2_cfg)
        if c_in == c_out and shortcut:
            self.shortcut = True
        else:
            self.shortcut = False
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

        if  self.quant is not None:
            if self.quant == "torch":
                self.skip_add = nn.quantized.FloatFunctional()

            #if self.quant == "quant":
            #    self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
            


    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        if self.shortcut:
            if self.quant == "torch":
                return self.skip_add.add( x, outputs)
            elif self.quant == "quant":
                outputs += self.residual_quantizer(x)
                return outputs 
            else:
                return  x + outputs
        else:
            return outputs
    

# 3. multiplt Residual convolutional block

class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, c_in, c_out, n=2, block=BottleRep, basic_block=Conv, quant = None , BottleRep_cfg_list=None):
        super().__init__()
        
        assert BottleRep_cfg_list is not None
        assert len(BottleRep_cfg_list) == n
        self.quant = quant
        print(" check ------------:", self.quant)
        self.conv1 = block(c_in, c_out, basic_block=basic_block, quant = quant, **BottleRep_cfg_list[0])

        
        #assert len(shortcut)==n
        self.block = nn.Sequential(*(block(c_in, c_out, basic_block=basic_block, quant = quant, **BottleRep_cfg_list[index+1]) for index in range(n - 1))) if n > 1 else None


    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x
    
    
# 4. Multi Brauch Block

class branch_block(nn.Module):
    def __init__(self, c_in, c_out, quant = None, n=1, e=0.5, concat=True, block=BottleRep, block_cfg = None, basic_block=Conv, cv1_cfg=None, cv2_cfg=None, cv3_cfg=None):  
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()

        assert block_cfg is not None
        assert cv1_cfg is not None
        assert cv2_cfg is not None
        assert cv3_cfg is not None
        
        
        c_ = int(c_out * e)  # hidden channels
        self.cv1 = basic_block(c_in, c_, **cv1_cfg)
        self.cv2 = basic_block(c_in, c_, **cv2_cfg)
        self.cv3 = basic_block(2 * c_, c_out,  **cv3_cfg)

        
        self.m = RepBlock(c_in=c_, c_out=c_, n=n, block=block, basic_block=basic_block, quant = quant, **block_cfg)
        self.concat = concat

        self.quant   = quant
        print(" check ------------:", self.quant)
        if  self.quant is not None:
            if self.quant == "torch" and self.concat:
                self.quant_cat = torch.nn.quantized.FloatFunctional()
    
        if not concat:
            self.cv3 = basic_block(c_, c_out, **cv3_cfg)

    def forward(self, x):
        if self.concat is True:
            if self.quant == "torch" :
                return self.cv3(self.quant_cat.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))
            else:
                return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))
        

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def parse_branch_block(cfg):
    cv1_kernel = cfg[0]
    cv2_kernel = cfg[1]
    cv2_kernel = cfg[2]
    branch_block_kernels = cfg[3]
    branch_block_skips   = cfg[4]
    branch_block_e = cfg[5]
    
    n = len(branch_block_kernels)//2
    assert n == len(branch_block_skips)

    cv1_cfg = {"kernel_size":cv1_kernel,"act":True, "bias" : True}
    cv2_cfg = {"kernel_size":cv2_kernel,"act":True, "bias" : True}
    cv3_cfg = {"kernel_size":cv2_kernel,"act":True, "bias" : True}
    
    BottleRep_cfg_list = []
    for i in range(n):
        conv1_cfg = {"kernel_size":branch_block_kernels[2*i],   "stride":1, "padding":None, "groups":1, "dilation":1, "act":True, "bias" : True}
        conv2_cfg = {"kernel_size":branch_block_kernels[2*i+1], "stride":1, "padding":None, "groups":1, "dilation":1, "act":True, "bias" : True}
        bottlerep1_cfg = {"shortcut":branch_block_skips[i], "conv1_cfg":conv1_cfg, "conv2_cfg":conv1_cfg}
        BottleRep_cfg_list.append(bottlerep1_cfg)
    RepBlock_cfg = {"BottleRep_cfg_list" : BottleRep_cfg_list}
    
    branch_block_cfg = {"e":branch_block_e, "n":n,  "concat":True, "block_cfg":RepBlock_cfg, "cv1_cfg":cv1_cfg, "cv2_cfg":cv2_cfg,"cv3_cfg":cv3_cfg}
    return branch_block_cfg



class Backbone(nn.Module):
    def __init__(self, config,
                 width_mult=0.5,
                 round_nearest=8,
                pre_img = False, quant = None, combine_style = "add"):

        super().__init__()

        input_channel = 64
        self.pre_img = pre_img
        self.quant   = quant
        self.combine_style = combine_style

        assert self.combine_style in ["add","cat","sub_cat","add_cat"]#,"sub_add"]
        
        print(" check ------------:", self.quant)
        if  self.quant is not None:
            if self.quant == "torch":
                self.skip_add = nn.quantized.FloatFunctional()
            #if self.quant == "quant":
            #    self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        if pre_img:
            print('adding pre_img layer...')
            # this is default
            self.pre_img_layer = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU())
            #)

        # this is also default
        features = [Conv(3, input_channel, kernel_size=3, stride=2)]


        if self.combine_style == "add":
            input_channel = input_channel
        elif  self.combine_style == "cat":
            input_channel = input_channel * 2
        elif self.combine_style == "sub_cat":
            input_channel = input_channel * 3
        elif self.combine_style == "add_cat":
             input_channel = input_channel * 3
        else :
            raise Exception("non valid combine_style")
    
        self.key_block = [True]
        all_channels = [input_channel]
        self.channels = [input_channel]
        # building inverted residual blocks
        # for op, layer_cfg in config:
        print(config)
        for op_name, layer_cfg in config:
            output_channel = layer_cfg[0]
            op_cfg         = layer_cfg[1]
            output_channel = _make_divisible(output_channel * width_mult, round_nearest)
            
            # parse the config for the operation
            
            #op_name        = str(op)[8:-2].replace('__main__.', '').split(".")[1]
            op = eval(op_name)
            
            if op_name == "Conv":
                stride = op_cfg[1]
                op_cfg = {"kernel_size":op_cfg[0], "stride":op_cfg[1], "padding":None, "groups":1, "dilation":1, "act":True, "bias" : True}
                features.append(op(c_in = input_channel, c_out = output_channel, **op_cfg))
            elif op_name == "branch_block":
                stride = 1
                op_cfg = parse_branch_block(op_cfg)
                features.append(op(c_in = input_channel, c_out = output_channel,  quant = quant, **op_cfg))
            else:
                assert 1==0
            
            
            
        
            #features.append(op(c_in = input_channel, c_out = output_channel, **op_cfg))
            input_channel = output_channel
            if stride == 2:
                self.key_block.append(True)
            else:
                self.key_block.append(False)
            all_channels.append(output_channel)
        #print(self.key_block)
        for i in range(len(self.key_block) - 1):
            if self.key_block[i + 1]:
                self.key_block[i] = True
                self.key_block[i + 1] = False
                self.channels.append(all_channels[i])

        self.key_block[-1] = True
        self.channels.append(all_channels[-1])
        #print('channels', self.channels)
        
        
        # make it nn.Sequential
        self.features = nn.ModuleList(features)
        
        print('len(self.features)', len(self.features))


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, inputs, pre_img=None, pre_hm=None):
        x = self.features[0](inputs)

        if pre_img is not None:
            if self.combine_style == "add":
                if self.quant == "torch":
                    x = self.skip_add.add(x, self.pre_img_layer(pre_img))
                elif self.quant == "quant":
                    pre_img = self.pre_img_layer(pre_img)
                    x += self.residual_quantizer(pre_img)
    
                else:
                    x = x + self.pre_img_layer(pre_img)
            elif  self.combine_style == "cat":
                x = torch.cat(( x, self.pre_img_layer(pre_img)), dim=1)

            elif self.combine_style == "sub_cat":
                x_pre_img = self.pre_img_layer(pre_img)
                sub_img = x - x_pre_img
                x = torch.cat(( x, x_pre_img, sub_img), dim=1)

            elif self.combine_style == "add_cat":
                x_pre_img = self.pre_img_layer(pre_img)
                add_img = x + x_pre_img
                x = torch.cat(( x, x_pre_img, add_img), dim=1)
            else :
                raise Exception("non valid combine_style")

            
            #x = x + self.pre_img_layer(pre_img)

        y = [x]
        for i in range(1, len(self.features)):
            x = self.features[i](x)
            # print('i, shape, is_key', i, x.shape, self.key_block[i])
            if self.key_block[i]:
                #print("-")
                y.append(x)
        return y
    
# ------------------ Neck ----------------------

# 【cin, cout, [1],[3],[1], up_ratio】

class BottleRep_neck(nn.Module):

    def __init__(self, c_in, config_list, quant = None):
        super().__init__()
        assert len(config_list)>=1
        self.quant = quant
        print(" check ------------:", self.quant)
        c_out  = c_in
        length = len(config_list)
        
        if length>1:
            self.shortcut = True
        else:
            self.shortcut = False
        conv_list = []
        for config in config_list:
            conv_list.append(Conv(c_in, c_out,**config))
        self.conv = nn.Sequential(*conv_list)
        if self.quant is not None:
            if self.quant == "torch":
                self.skip_add = nn.quantized.FloatFunctional()
            #if self.quant == "quant":
            #    self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
                
    def forward(self, x):
        if self.shortcut:
            if self.quant == "torch":
                return self.skip_add.add(x, self.conv(x))
            elif self.quant == "quant":
                outputs = self.conv(x)
                outputs += self.residual_quantizer(x)
                return outputs
            else:
                return x + self.conv(x)
        else:
            return self.conv(x)
        
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

        
class BottleRep_neck_IDAUp(nn.Module):
    def __init__(self, c_in, c_out, configs, quant = None):
        # configs  [1,[3,3,3],2,[3,3,3]]
        super(BottleRep_neck_IDAUp, self).__init__()
        self.quant = quant

        if self.quant is not None:
            if self.quant == "torch":
                self.skip_add = nn.quantized.FloatFunctional()
            #if self.quant == "quant":
            #    self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        print(" check ------------:", self.quant)
        kernel_project = configs[0]
        kernel_preject_congfig = {"kernel_size":kernel_project,   "stride":1, "padding":None, "groups":1, "dilation":1, "act":True, "bias" : True}
        self.project =  Conv(c_in, c_out, **kernel_preject_congfig)
        
        
        
        kernel_feature = configs[1]
        assert isinstance(kernel_feature, list)
        if len(kernel_feature) > 0:
            kernel_feature_config_list = []
            for i in kernel_feature:
                kernel_feature_config_list.append({"kernel_size":i,   "stride":1, "padding":None, "groups":1, "dilation":1, "act":True, "bias" : True})
            self.feature = BottleRep_neck(c_out,kernel_feature_config_list, quant)
        else:
            self.feature = None
            
        up_ratio      = configs[2]
        self.up = nn.ConvTranspose2d(c_out, c_out, 
                                     up_ratio * 2, stride=up_ratio, 
                                     padding=up_ratio // 2, output_padding=0,
                                     groups=c_out, bias=False) # 变大wh
        fill_up_weights(self.up)
        
        
        
        kernel_node    = configs[3]
        assert isinstance(kernel_node, list)
        assert len(kernel_node)>0
        kernel_node_config_list = []
        for i in kernel_node:
            kernel_node_config_list.append({"kernel_size":i,   "stride":1, "padding":None, "groups":1, "dilation":1, "act":True, "bias" : True})
        self.node = BottleRep_neck(c_out,kernel_node_config_list, quant)
        
                 
    #def upsample_enable_quant(self, num_bits, calib_method):              
    #    conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)
    #    self.up_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
    #    self._QUANT = True

    
    def forward(self, x1,x2):
        # x1 has more downration 2 times than x2

        x2 = self.project(x2)
        if self.feature is not None:
            x2 = self.feature(x2)


        x2 = self.up(x2)
        if self.quant == "quant" and hasattr(self, '_QUANT') and self._QUANT is True:
            x2 = self.up_quant(x2)
        
        
        if self.quant == "torch":
            return self.node(self.skip_add.add(x1, x2))

        elif self.quant == "quant":
            x2 += self.residual_quantizer(x1) 
            return self.node(x2)
        else:
            return self.node(x1+x2)
        
        
class Neck(nn.Module):
    def __init__(self,  channels, configs, quant=None):
        super().__init__()

        self.channels = channels # [32, 48, 64, 128, 256, 512]
        assert len(self.channels) == 4
        assert len(configs) == len(self.channels)-1+2
        self.quant = quant
        print(" check ------------:", self.quant)
        #down_ratio = 4
        config_list = configs[0]
        self.up_level_1 = []
        for i, cfg in enumerate(config_list):
            self.up_level_1.append(BottleRep_neck_IDAUp(c_in=self.channels[i+1], c_out=self.channels[i],configs=cfg, quant=quant))
        self.up_level_1 = nn.ModuleList(self.up_level_1)
          
        config_list = configs[1]
        self.up_level_2 = []
        for i, cfg in enumerate(config_list):
            self.up_level_2.append(BottleRep_neck_IDAUp(c_in=self.channels[i+1], c_out=self.channels[i],configs=cfg, quant=quant))
        self.up_level_2 = nn.ModuleList(self.up_level_2)
        
        config_list = configs[2]    
        self.up_level_3 = []
        for i, cfg in enumerate(config_list):
            self.up_level_3.append(BottleRep_neck_IDAUp(c_in=self.channels[i+1], c_out=self.channels[i],configs=cfg, quant=quant))
        self.up_level_3 = nn.ModuleList(self.up_level_3)
        
        config = configs[3]
        self.up_level_4 = BottleRep_neck_IDAUp(c_in=self.channels[1], c_out=self.channels[0],configs=config, quant=quant)
        
        
        config = configs[4]
        self.up_level_5 = BottleRep_neck_IDAUp(c_in=self.channels[2], c_out=self.channels[0],configs=config, quant=quant)

    def forward(self, x1, x2, x3, x4):
        x_list = [x1, x2, x3, x4]
        out_level_1 = []
        for i,op in enumerate(self.up_level_1):
            out_level_1.append(op(x_list[i],x_list[i+1]))
            
        out_level_2 = []
        for i,op in enumerate(self.up_level_2):
            out_level_2.append(op(out_level_1[i],out_level_1[i+1]))
            
            
        out_level_3 = []
        for i,op in enumerate(self.up_level_3):
            out_level_3.append(op(out_level_2[i],out_level_2[i+1]))
            
        out = self.up_level_4(out_level_3[-1],out_level_2[-1])
        out = self.up_level_5(out, out_level_1[-1])

        return out
    
# -----------supernet -------------------------

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class SuperNetNetwork(nn.Module):
    def __init__(self,  heads, nectwork_seed = 1, width_mult=0.5, opt = None):
        super(SuperNetNetwork, self).__init__()
        
        self.opt = opt
        self.quant = opt.quant
        print(" check ------------:", self.quant)
        if self.quant == "torch":
            
            self.input_quant = torch.quantization.QuantStub()
            self.output_quant = torch.quantization.DeQuantStub()
    
        nectwork_config = generate_one_random_config_1(seed = nectwork_seed)
        #nectwork_config = generate_one_random_config_small(seed = nectwork_seed)
        assert nectwork_config["backbone"] is not None
        assert nectwork_config["neck"] is not None
        assert nectwork_config["head"] is not None
        
        backbone_config = nectwork_config["backbone"]
        self.backbone = Backbone(backbone_config,width_mult,pre_img = opt.pre_img, quant =self.quant, combine_style = opt.combine_style)
        channels = self.backbone.channels[2:]
        
        neck_config = nectwork_config["neck"]
        self.neck = Neck(channels, neck_config, self.quant)

        last_channel = channels[0]
        self.heads = heads
        head_kernel = 3
        prior_bias = -4.6
        head_configs = nectwork_config["head"]
        for head in self.heads: # {'hm': num_cls, 'reg': 2, 'wh': 2}
            classes = self.heads[head]
            head_config = head_configs[head]
            
            out = nn.Conv2d(last_channel, classes, kernel_size=1, stride=1, padding=0, bias=True)
            
            if len(head_config)>0:
                config_list = []
                for i in head_config:
                    config_list.append( {"kernel_size":i,   "stride":1, "padding":None, "groups":1, "dilation":1, "act":True, "bias" : True})

                
                convs = BottleRep_neck(last_channel,config_list, self.quant)
            
            else:
                convs = None
            if convs is not None:
                fc = nn.Sequential(convs,out)
            else:
                fc = nn.Sequential(out)
                
            if 'hm' in head:
                fc[-1].bias.data.fill_(prior_bias)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
            
            
    def forward(self, x, pre_img=None, pre_hm=None):
        if self.quant == "torch":
            x = self.input_quant(x)
            pre_img = self.input_quant(pre_img)
    
        y = self.backbone(x, pre_img, pre_hm)
        # print(y[2].shape)
        # print(y[3].shape)
        # print(y[4].shape)
        # print(y[5].shape)
        feats = self.neck(*y[2:])
        out = []
        
        if self.opt.model_output_list:
            z = []

            for head in sorted(self.heads):
                #z.append(self.__getattr__(head)(feats[s]))
                z_temp = self.__getattr__(head)(feats)
                if self.quant == "torch":
                    z_temp = self.output_quant(z_temp)
                z.append(z_temp)
            out.append(z)
        else:
            z = {}
            for head in sorted(self.heads):
                z_temp = self.__getattr__(head)(feats)
                if self.quant == "torch":
                    z_temp = self.output_quant(z_temp)
                z[head] = z_temp

            out.append(z)
            
        # print("############################",out[-1].keys())
        # print(out[-1]["hm"])
        return out
    
    
def generate_one_random_config_1(seed, kernel_list = [1,3,5], p=[0.4,0.4,0.2]):
    from numpy import random
    if p is None:
        p = [1/len(kernel_list) for _ in range(len(kernel_list))]

    random.seed(seed)
    back_config = []
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [96,    [ks,      1]]])
    # ===============================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [128,    [ks,      2]]])#  1-P2/4
    # ----------------------------------
    length = random.choice([1,2])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]

    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [128,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # ================================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [256,    [ks,      2]]])
    # ---------------------------------------
    length = random.choice([1,2,3,4])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]
    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [256,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # ================================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [512,    [ks,      2]]])
    # -------------------------------------------
    length = random.choice([1,2,3,4,5,6])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]
    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [256,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # ================================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [1024,    [ks,      2]]])
    # -------------------------------------------
    length = random.choice([1,2])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]
    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [1024,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # -------------------------------------------
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [1024,    [ks,      1]]])
    

    
    # =============NECK ================================
    neck_config = []
    for i in range(3):
        temp_list = []
        for j in range(3-i):
            temp_list.append([int(random.choice(kernel_list,p=p)),[int(random.choice(kernel_list,p=p))  for _ in range(random.choice([0,1,2]))],2,  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1,2]))]])
        neck_config.append(temp_list)
    

    temp_list = [int(random.choice(kernel_list,p=p)),[int(random.choice(kernel_list,p=p))  for _ in range(random.choice([0,1,2]))],2,  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1,2]))]]
    neck_config.append(temp_list)
    

    temp_list = [int(random.choice(kernel_list,p=p)),[int(random.choice(kernel_list,p=p))  for _ in range(random.choice([0,1,2]))],4,  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1,2]))]]
    neck_config.append(temp_list)

    
    # ==============HEAD ============================
    
    head_config ={
        'hm':  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1,2]))],
        'reg': [int(random.choice(kernel_list,p=p)) for _ in range(random.choice([1,2]))],
        'wh':  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1,2]))],
        'tracking': [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1,2]))]
    }

    nectwork_config = {"backbone":back_config, "neck":neck_config, "head":head_config}
    return nectwork_config


def generate_one_random_config_small(seed, kernel_list = [1,3,5], p=[0.4,0.4,0.2]):
    from numpy import random
    if p is None:
        p = [1/len(kernel_list) for _ in range(len(kernel_list))]

    random.seed(seed)
    back_config = []
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [96,    [ks,      1]]])
    # ===============================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [128,    [ks,      2]]])#  1-P2/4
    # ----------------------------------
    length = random.choice([1])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]

    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [128,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # ================================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [256,    [ks,      2]]])
    # ---------------------------------------
    length = random.choice([1])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]
    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [256,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # ================================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [512,    [ks,      2]]])
    # -------------------------------------------
    length = random.choice([1])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]
    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [256,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # ================================================
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [1024,    [ks,      2]]])
    # -------------------------------------------
    length = random.choice([1])
    ks_list = [ int(random.choice(kernel_list,p=p)) for _ in range(2*length)]
    sk_list = [ random.choice([0,1])   for _ in range(length)]
    back_config.append(["branch_block", [1024,[int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),int(random.choice(kernel_list,p=p)),ks_list,sk_list,0.5]]])
    # -------------------------------------------
    ks = int(random.choice(kernel_list,p=p))
    back_config.append(["Conv",    [1024,    [ks,      1]]])
    

    
    # =============NECK ================================
    neck_config = []
    for i in range(3):
        temp_list = []
        for j in range(3-i):
            temp_list.append([int(random.choice(kernel_list,p=p)),[int(random.choice(kernel_list,p=p))  for _ in range(random.choice([0]))],2,  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1]))]])
        neck_config.append(temp_list)
    

    temp_list = [int(random.choice(kernel_list,p=p)),[int(random.choice(kernel_list,p=p))  for _ in range(random.choice([0]))],2,  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1]))]]
    neck_config.append(temp_list)
    

    temp_list = [int(random.choice(kernel_list,p=p)),[int(random.choice(kernel_list,p=p))  for _ in range(random.choice([0]))],4,  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1]))]]
    neck_config.append(temp_list)

    
    # ==============HEAD ============================
    
    head_config ={
        'hm':  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1]))],
        'reg': [int(random.choice(kernel_list,p=p)) for _ in range(random.choice([1]))],
        'wh':  [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1]))],
        'tracking': [int(random.choice(kernel_list,p=p))  for _ in range(random.choice([1]))]
    }

    nectwork_config = {"backbone":back_config, "neck":neck_config, "head":head_config}
    return nectwork_config