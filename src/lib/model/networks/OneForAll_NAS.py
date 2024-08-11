import warnings
warnings.filterwarnings("ignore")

#from pytorch_quantization import nn as quant_nn
#from pytorch_quantization.tensor_quant import QuantDescriptor

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import random
import copy
import math

# -------------- convert/compress the model--------------------------

def rep_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


# ------------------------ basic Conv Batchnormalization Block ----------------------
# Define a helper function to create a convolutional layer followed by batch normalization
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    # Use nn.Sequential to combine multiple layers into a single module
    result = nn.Sequential()
    # Add a convolutional layer
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, 
                                        out_channels=out_channels,
                                        kernel_size=kernel_size, 
                                        stride=stride, 
                                        padding=padding, 
                                        groups=groups, 
                                        bias=False))
    # Add a batch normalization layer
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


# ------------------------ Rep Convolutional Batch normalization Block -------------------------


class NAS_REP_ConvBNBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        max_kernel_size,
        stride, 
        deploy = False,
        verbose = False,
        scaling_factor = 0.5,
        run_time_depth_index = 0):
        """
        Initialize the block with given parameters.
        in_channels               : number of input channels
        out_channels              : number of output channels
        max_kernel_size           : maximum kernel size
        stride                    : stride for the convolution
        deploy                    : if True, deploy the model for inference
        verbose                   : if True, print log messages
        scaling_factor            : scaling factor for combining the outputs
        run_time_depth_index      : 这是标记这个conv属于progressive中的哪个stage，如果， 默认都是零
        """

        super(NAS_REP_ConvBNBlock, self).__init__()
        self.in_channels            =  in_channels
        self.out_channels           =  out_channels
        self.max_kernel_size        =  max_kernel_size
        self.stride                 =  stride
        self.verbose                =  verbose
        self.deploy                 =  deploy
        self.scaling_factor         =  scaling_factor # 因为是所有的输出就要相加，所以当kernel过大的时候，可能需要一个scaling
        self.run_time_depth_index   = run_time_depth_index
        self.current_run_time_depth = None

        assert max_kernel_size>=1
        assert max_kernel_size%2==1

        self.out_channels   =  out_channels
        self.in_channels    =  in_channels
        self.kernel_list    =  [0] + list(range(1,max_kernel_size+1,2)) # kernel list是永远包括了零的，但是有个情况就是 stride==2的时候，不包含0。
        # 这个反应在可以选择的 kernel_index_range

        self.active_kernel_index = len(self.kernel_list)-1 

        if self.stride == 2 or out_channels != in_channels:
            self.kernel_index_range = list(range(1,self.active_kernel_index+1))
        else:
            self.kernel_index_range = list(range(0,self.active_kernel_index+1))

        self.nonlinearity = nn.ReLU()
        

        # -------------------- Identity -Skip Connection  ------------------------------
        # ==================== TODO Identidy              ======================================
        if out_channels == in_channels and stride == 1:
            setattr(self, f"branch_kernel_0", nn.BatchNorm2d(num_features=in_channels))
        else:
            setattr(self, f"branch_kernel_0", None)


        # -------------------- convolutional with Batchnormalizition with kernel >= 1------------------------
        for k in self.kernel_list[1:]:
            setattr(self, f"branch_kernel_{k}", conv_bn(in_channels, out_channels, k, stride, k//2))


    def log(self, message):
        if self.verbose:
            print(message)

    def forward(self, inputs):
        
        if self.active_kernel_index in self.kernel_index_range:
            if self.current_run_time_depth is None or self.current_run_time_depth>=self.run_time_depth_index:
                if hasattr(self, 'conv_reparam'):
                    return self.nonlinearity(self.conv_reparam(inputs))
                
                else:
                    if getattr(self, f"branch_kernel_0") is not None:
                        self.log("using branch_kernel_0")
                        output = self.scaling_factor * getattr(self, f"branch_kernel_0")(inputs)
        
                        index = 1
                    else:
                        self.log("using branch_kernel_1")
                        output = self.scaling_factor * getattr(self, f"branch_kernel_1")(inputs)
                        index = 2  
                    
                    for k in self.kernel_list[index:self.active_kernel_index+1]:
                        self.log(f"using branch_kernel_{k}")
                        output += self.scaling_factor * getattr(self, f"branch_kernel_{k}")(inputs)
        
        
                    return self.nonlinearity(output)
            else:
                return inputs
            
        else:
            return inputs


    def _fuse_bn_tensor(self, branch):

        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // 1
                if self.active_kernel_index>0:
                    kernel_value = np.zeros((self.in_channels, input_dim, self.kernel_list[self.active_kernel_index], self.kernel_list[self.active_kernel_index]), dtype=np.float32)
                else:
                    kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_list[self.active_kernel_index]//2, self.kernel_list[self.active_kernel_index]//2] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


    def get_equivalent_kernel_bias(self):
 
        kernel, bias = self._fuse_bn_tensor(getattr(self, f"branch_kernel_0"))
    
        if kernel is not None:
            kernel = kernel * self.scaling_factor
            bias   = bias   * self.scaling_factor

        
        for k in self.kernel_list[1:self.active_kernel_index+1]:
            temp_kernel, temp_bias = self._fuse_bn_tensor(getattr(self, f"branch_kernel_{k}"))
            pld = int((self.kernel_list[self.active_kernel_index]-k)/2)
            if pld > 0 :
                temp_kernel = torch.nn.functional.pad(temp_kernel, [pld, pld, pld, pld])

            kernel = temp_kernel  * self.scaling_factor + kernel
            bias   = temp_bias    * self.scaling_factor + bias
        

        return kernel, bias


    def set_active_sample_net_param(self):

        for kernel in self.kernel_list:
            if getattr(self, f"branch_kernel_{kernel}") is not None:
                for param in getattr(self, f"branch_kernel_{kernel}").parameters():
                    param.requires_grad = False
        

        kernel = self.kernel_list[self.active_kernel_index]
        for param in getattr(self, f"branch_kernel_{kernel}").parameters():
                param.requires_grad = True

    def set_active_sample_all_net_param(self):

        for kernel in self.kernel_list:
            if getattr(self, f"branch_kernel_{kernel}") is not None:
                for param in getattr(self, f"branch_kernel_{kernel}").parameters():
                    param.requires_grad = False
        
        for kernel in self.kernel_list[:self.active_kernel_index+1]:
            if getattr(self, f"branch_kernel_{kernel}") is not None:
                for param in getattr(self, f"branch_kernel_{kernel}").parameters():
                        param.requires_grad = True

    def reset_active_net_param(self):

        for kernel in self.kernel_list:
            if getattr(self, f"branch_kernel_{kernel}") is not None:
                for param in getattr(self, f"branch_kernel_{kernel}").parameters():
                    param.requires_grad = True
        self.active_kernel_index = len(self.kernel_list)-1 
        self.current_run_time_depth = None
        

    def generate_active_convs(self):
        self.active_kernel_index = random.choice(self.kernel_index_range)

    # def set_active_convs(self, conv_index):
    #     #assert conv_index in self.kernel_index_range
    #     self.active_kernel_index = conv_index
        
    # def set_current_run_time_depth(self, current_run_time_depth):
    #     self.current_run_time_depth = current_run_time_depth

    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):


        for kernel in self.kernel_list:
            if getattr(self, f"branch_kernel_{kernel}") is not None:
                for param in getattr(self, f"branch_kernel_{kernel}").parameters():
                    param.requires_grad = False

        
        if kernel_index is None:
            assert random == True
        else:
            assert random == False

        if random:
            assert kernel_index is None
        else:
            assert kernel_index is not None
            
        self.current_run_time_depth = current_run_time_depth

        if current_run_time_depth<self.run_time_depth_index:
            kernel_index = 0
        elif current_run_time_depth == self.run_time_depth_index:
            active_net = True
        else:
            if not all_layers and not random:
                kernel_index = len(self.kernel_list)-1 

            # --------?----------
            if random or all_layers:
                if not active_all_sub_net:
                    active_net = True


        if kernel_index is not None:
            if kernel_index == 0:
                if self.stride == 2 or self.out_channels != self.in_channels:
                    kernel_index = 1
                
            self.active_kernel_index = kernel_index
        else:
            self.generate_active_convs()


        if current_run_time_depth == self.run_time_depth_index:
            if active_net:
                self.set_active_sample_net_param()

            if active_all_sub_net:
                self.set_active_sample_all_net_param()
        elif current_run_time_depth>self.run_time_depth_index:
            if all_layers or random:
                if active_net:
                    self.set_active_sample_net_param()
    
                if active_all_sub_net:
                    self.set_active_sample_all_net_param() 




        
    def switch_to_deploy(self, only_for_weight=False):
        

            
        if hasattr(self, 'conv_reparam'):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
            
        if only_for_weight:
            return kernel, bias
        if self.active_kernel_index>0:
            self.conv_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=self.kernel_list[self.active_kernel_index], stride=self.stride,
                                         padding=self.kernel_list[self.active_kernel_index]//2, dilation=1, groups=1, bias=True)
        else:
            
            self.conv_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=1, stride=self.stride,
                                         padding=1//2, dilation=1, groups=1, bias=True)
        

        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias
        
        for para in self.parameters():
            para.detach_()
            
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

        for k in self.kernel_list:
            if hasattr(self, f'branch_kernel_{k}'):
                self.__delattr__(f'branch_kernel_{k}')
        self.deploy = True

















# ----------------- BackBone ----------------

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



# 2. Residual convolutional block ---> 2 convolutional + skip connection

class BottleRep(nn.Module):

    def __init__(
        self, 
        c_in, 
        c_out, 
        max_kernel_size, 
        stride= 1, 
        number_blocks = 2, 
        verbose = False, 
        scaling_factor = 0.5, 
        run_time_depth_index = 0, 
        keep_same_run_time = True, 
        skip_connection=True
    ):
        super().__init__()
        self.c_in                 = c_in
        self.c_out                = c_out
        self.keep_same_run_time   = keep_same_run_time
        self.skip_connection      = skip_connection
        self.run_time_depth_index = run_time_depth_index
        self.number_blocks        = number_blocks

        if not keep_same_run_time:
            self.run_time_depth_index_list = list(range(run_time_depth_index,run_time_depth_index+number_blocks)) 
        else:
            self.run_time_depth_index_list = None


        self.current_run_time_depth = None
        # because of the residual connection
        assert c_in   == c_out
        assert stride == 1
        
        self.number_blocks =number_blocks
        for b in range(number_blocks):
            setattr(self, f"conv_block_{b}", NAS_REP_ConvBNBlock(
                c_in,
                c_out,
                max_kernel_size,
                stride, 
                verbose=verbose,
                scaling_factor=scaling_factor, 
                run_time_depth_index = run_time_depth_index))
            if not self.keep_same_run_time:
                run_time_depth_index = run_time_depth_index + 1
            if b == 0 :
                c_in = c_out

    # def set_same_active_conv_all_blocks(self, conv_index):
    #     for b in range(self.number_blocks):
    #         getattr(self, f"conv_block_{b}").set_active_convs(conv_index)
            
    # def activate_sampled_blocks(self):
    #     for b in range(self.number_blocks):
    #         getattr(self, f"conv_block_{b}").set_active_net_param()
            
    # def set_random_active_conv(self):
    #     for b in range(self.number_blocks):
    #         getattr(self, f"conv_block_{b}").generate_active_convs()


    def reset_active_depth_and_conv(self):
        for b in range(self.number_blocks):
            getattr(self, f"conv_block_{b}").reset_active_net_param()
        self.current_run_time_depth = None



    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):


        self.current_run_time_depth = current_run_time_depth
        
        for b in range(self.number_blocks):
            getattr(self, f"conv_block_{b}").set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        


    def forward(self, x):

        if self.current_run_time_depth is None or self.current_run_time_depth>=self.run_time_depth_index:

            residual = x
            
            for b in range(self.number_blocks):
    
                x = getattr(self, f"conv_block_{b}")(x)

            if self.skip_connection:
    
                return x + residual
            else:
                return x
        else:
            return x




# 3. multiplt Residual convolutional block

class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(
        self,   
        c_in,   
        c_out,  
        max_kernel_size,  
        stride=1, 
        number_BottleRep = 4, 
        number_blocks_list = None, 
        verbose=False, 
        scaling_factor=0.5, 
        run_time_depth_index=0
    ):
        super().__init__()

        self.c_in                      = c_in
        self.c_out                     = c_out
        self.verbose                   = verbose
        assert c_in == c_out
        assert stride == 1 
        self.max_kernel_size           = max_kernel_size
        self.number_BottleRep          = number_BottleRep
        
        self.run_time_depth_index      = run_time_depth_index

        
        self.run_time_depth_index_list = list(range(run_time_depth_index,run_time_depth_index+number_BottleRep)) 


        self.current_run_time_depth    = None

        if number_blocks_list is None:
            self.number_blocks_list = [2] * number_BottleRep
        else:
            assert len(number_blocks_list) == number_BottleRep
            self.number_blocks_list = number_blocks_list
        
        self.log("number_blocks_list" + str(self.number_blocks_list))

        run_time_depth_start_index = run_time_depth_index
        for b in range(number_BottleRep):
            setattr(self, f"BottleRep_{b}", BottleRep(c_in,
                                                      c_out,
                                                      max_kernel_size,
                                                      stride, 
                                                      number_blocks = self.number_blocks_list[b],
                                                      verbose = verbose,
                                                      scaling_factor = scaling_factor,
                                                      run_time_depth_index = run_time_depth_start_index,
                                                      keep_same_run_time = True, 
                                                      skip_connection=True
                                                     )
                   )


        
            run_time_depth_start_index = run_time_depth_start_index + 1

        
    def reset_active_depth_and_conv(self):


        for b in list(range(self.number_BottleRep)):

            getattr(self, f"BottleRep_{b}").reset_active_depth_and_conv()

        self.current_run_time_depth = None



    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):


        self.current_run_time_depth = current_run_time_depth
        for b in range(self.number_BottleRep):
                getattr(self, f"BottleRep_{b}").set_current_runtime_depth_and_kernel(
                    current_run_time_depth, 
                    kernel_index, 
                    
                    random,
                    all_layers,

                    active_net,
                    active_all_sub_net
                )
        

    
    def forward(self, x):

        if self.current_run_time_depth is None or self.current_run_time_depth >= max(self.run_time_depth_index_list): 

            for b in list(range(self.number_BottleRep)):
    
                x = getattr(self, f"BottleRep_{b}")(x)
            return x

        elif self.current_run_time_depth in self.run_time_depth_index_list:
            for b in list(range(self.number_BottleRep))[:self.run_time_depth_index_list.index(self.current_run_time_depth)+1]:
    
                x = getattr(self, f"BottleRep_{b}")(x)

            return x
        else:
            return x
        

    def log(self, message):
        if self.verbose:
            print(message)





class branch_block(nn.Module):
    def __init__(self, 
                 c_in,   
                 c_out,  
                 max_kernel_size,   
                 stride=1, 
                 number_BottleRep=2,  
                 number_blocks_list = None, 
                 e=0.5, 
                 skip_connection=True,
                 verbose = False,
                 scaling_factor = 0.5,
                 run_time_depth_index=0):  

        super().__init__()
        
        assert stride == 1
        assert c_in == c_out 
        self.c_in                        = c_in
        self.c_out                       = c_out
        self.skip_connection             = skip_connection
        self.max_kernel_size             = max_kernel_size
        c_                               = int(c_out * e)  # hidden channels
        self.c_                          = c_
        self.run_time_depth_index        = run_time_depth_index
        self.run_time_depth_index_list   = list(range(run_time_depth_index,run_time_depth_index+number_BottleRep+1)) 
        self.current_run_time_depth      = None



        
        if self.skip_connection : 
        
            self.conv_block_reduction   = NAS_REP_ConvBNBlock(
                c_in, 
                c_,   
                max_kernel_size,
                stride,
                verbose=verbose,
                scaling_factor=scaling_factor, 
                run_time_depth_index = run_time_depth_index)

            
            self.RepBlock               = RepBlock( 
                c_in=c_, 
                c_out=c_,  
                max_kernel_size = max_kernel_size,
                stride=stride, 
                number_BottleRep=number_BottleRep ,
                number_blocks_list= number_blocks_list, 
                verbose=verbose, 
                scaling_factor = scaling_factor, 
                run_time_depth_index=run_time_depth_index+1)
            
            self.conv_block_skip        = NAS_REP_ConvBNBlock(
                c_in, 
                c_,   
                max_kernel_size,
                stride,
                verbose=verbose,
                scaling_factor=scaling_factor, 
                run_time_depth_index = run_time_depth_index)
            
            self.conv_block_backmapping = NAS_REP_ConvBNBlock(
                2*c_, 
                c_out,
                max_kernel_size,
                stride,
                verbose=verbose,
                scaling_factor=scaling_factor, 
                run_time_depth_index = run_time_depth_index)
        else:
            
            self.conv_block_reduction   = NAS_REP_ConvBNBlock(
                c_in, 
                c_,   
                max_kernel_size,
                stride,
                verbose=verbose,
                scaling_factor=scaling_factor,  
                run_time_depth_index = run_time_depth_index)
            
            self.RepBlock               = RepBlock(
                c_in=c_, 
                c_out=c_,  
                max_kernel_size = max_kernel_size,
                stride=stride, 
                number_BottleRep=number_BottleRep ,
                number_blocks_list= number_blocks_list, 
                verbose=verbose, 
                scaling_factor = scaling_factor, 
                run_time_depth_index=run_time_depth_index+1)
            
            self.conv_block_skip        = None
            
            self.conv_block_backmapping = NAS_REP_ConvBNBlock(
                c_, 
                c_out,
                max_kernel_size,
                stride,
                verbose=verbose,
                scaling_factor=scaling_factor,  
                run_time_depth_index = run_time_depth_index)
        


    def forward(self, x):
        if self.current_run_time_depth is None or self.current_run_time_depth>=min(self.run_time_depth_index_list):
            if self.skip_connection :

                return self.conv_block_backmapping(torch.cat((self.RepBlock(self.conv_block_reduction(x)), self.conv_block_skip(x)), dim=1))
            else:
                return self.conv_block_backmapping(self.RepBlock(self.conv_block_reduction(x)))
        else:
            return x



    def reset_active_depth_and_conv(self):
        self.conv_block_reduction.reset_active_net_param()
        self.conv_block_backmapping.reset_active_net_param()
        
        if self.skip_connection: 
            self.conv_block_skip.reset_active_net_param()

        self.RepBlock.reset_active_depth_and_conv()
            
        self.current_run_time_depth = None
        
    

    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):


        self.current_run_time_depth = current_run_time_depth



        self.conv_block_reduction.set_current_runtime_depth_and_kernel(current_run_time_depth, kernel_index,  random, all_layers, active_net, active_all_sub_net)
        self.conv_block_backmapping.set_current_runtime_depth_and_kernel(current_run_time_depth, kernel_index,  random, all_layers, active_net, active_all_sub_net)
        if self.skip_connection:
            self.conv_block_skip.set_current_runtime_depth_and_kernel(current_run_time_depth, kernel_index,  random, all_layers, active_net, active_all_sub_net)
        
        self.RepBlock.set_current_runtime_depth_and_kernel(current_run_time_depth, kernel_index,  random, all_layers, active_net, active_all_sub_net)


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

class Backbone(nn.Module):
    def __init__(self, 
                 config,
                 max_kernel_size,
                 width_mult=0.5,
                 round_nearest=8,
                 pre_img = False,
                 combine_style = "add",
                 verbose = False,
                 scaling_factor = 0.5,
                 skip_connection = True
                 ):

        super().__init__()

        input_channel                       = 64
        self.pre_img                        = pre_img
        self.combine_style                  = combine_style
        self.max_kernel_size                = max_kernel_size

        
        assert self.combine_style in ["add","cat","sub_cat","add_cat"]

        input_channel                       = _make_divisible(input_channel * width_mult, round_nearest)

        if pre_img:
            #print('adding pre_img layer...with input_channel',input_channel)
            self.pre_img_layer = NAS_REP_ConvBNBlock(
                3,
                input_channel,
                max_kernel_size, 
                stride=2,
                verbose = verbose,
                scaling_factor = scaling_factor,
                run_time_depth_index = 0
            )
            

        # this is also default
        features = [NAS_REP_ConvBNBlock(
            3, 
            input_channel, 
            max_kernel_size, 
            stride=2,
            verbose = verbose,
            scaling_factor = scaling_factor,
            run_time_depth_index = 0
        )]


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

        for op_name, layer_cfg in config:

            output_channel = layer_cfg[0]

            output_channel = _make_divisible(output_channel * width_mult, round_nearest)
            
            op = eval(op_name)
            
            if op_name == "NAS_REP_ConvBNBlock":
                stride = layer_cfg[1]
                run_time_depth_index = layer_cfg[2]

              
                features.append(op(
                    in_channels         = input_channel, 
                    out_channels        = output_channel, 
                    max_kernel_size     = max_kernel_size, 
                    stride              = stride,
                    verbose             = verbose,
                    scaling_factor      = scaling_factor,
                    run_time_depth_index= run_time_depth_index
                    )
                )
            elif op_name == "branch_block":
                n = layer_cfg[1]
                stride = 1
                run_time_depth_index = layer_cfg[2]
                features.append(op(
                    c_in                = input_channel, 
                    c_out               = output_channel,
                    max_kernel_size     = max_kernel_size,
                    stride              = 1,
                    number_BottleRep    = n,
                    e                   = 0.5,
                    skip_connection     = skip_connection,
                    verbose             = verbose,
                    scaling_factor      = scaling_factor,
                    run_time_depth_index= run_time_depth_index   
                ))


            else:
                assert 1==0
            
            input_channel = output_channel
            if stride == 2:
                self.key_block.append(True)
            else:
                self.key_block.append(False)
                
            all_channels.append(output_channel)

        for i in range(len(self.key_block) - 1):
            if self.key_block[i + 1]:
                self.key_block[i] = True
                self.key_block[i + 1] = False
                self.channels.append(all_channels[i])

        self.key_block[-1] = True
        self.channels.append(all_channels[-1])

        self.features = nn.ModuleList(features)
        

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


    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):
        if self.pre_img:
            self.pre_img_layer.set_current_runtime_depth_and_kernel(current_run_time_depth,  kernel_index, random, all_layers, active_net, active_all_sub_net)

        for layer in self.features:
            layer.set_current_runtime_depth_and_kernel(current_run_time_depth,  kernel_index, random, all_layers, active_net, active_all_sub_net)

                    
    def random_set_current_runtime_depth_and_kernel(self, 

                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):
        

        if self.pre_img:
            self.pre_img_layer.set_current_runtime_depth_and_kernel(
                0,  
                None, 
                True, 
                all_layers, 
                active_net, 
                active_all_sub_net)


        for layer in self.features:

            if type(layer).__name__  == "NAS_REP_ConvBNBlock":
                assert layer.run_time_depth_index == 0
                layer.set_current_runtime_depth_and_kernel(
                    0,  
                    None, 
                    True, 
                    all_layers, 
                    active_net, 
                    active_all_sub_net)
            else:
                assert type(layer).__name__  == "branch_block"
                random_run_depth = random.choice(layer.run_time_depth_index_list) 
                layer.set_current_runtime_depth_and_kernel(
                    random_run_depth,  
                    None, 
                    True, 
                    all_layers,
                    active_net, 
                    active_all_sub_net)

    def reset_active_depth_and_conv(self):

        if self.pre_img:
            self.pre_img_layer.reset_active_net_param()

        for layer in self.features:

            if type(layer).__name__  == "NAS_REP_ConvBNBlock":
                layer.reset_active_net_param()
            else:
                assert type(layer).__name__  == "branch_block"
                layer.reset_active_depth_and_conv()

    def forward(self, inputs, pre_img=None, pre_hm=None):
        # ------------!!!!!!!!!!!!!!!-------------
        x = self.features[0](inputs)

        if pre_img is not None:
            if self.combine_style == "add":
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

            if self.key_block[i]:

                y.append(x)
        return y
    

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
    def __init__(self, c_in, c_out, max_kernel_size,configs,verbose=False, scaling_factor = 0.5, run_time_depth_index=0):
        # configs  [1,[3,3,3],2,[3,3,3]]
        # [3, [5], 2, [3]]
        super(BottleRep_neck_IDAUp, self).__init__()

        self.c_in                 = c_in
        self.c_out                = c_out
        self.max_kernel_size      = max_kernel_size
        self.scaling_factor       = scaling_factor
        self.run_time_depth_index = run_time_depth_index


        self.project =  NAS_REP_ConvBNBlock(
            c_in,
            c_out,
            max_kernel_size, 
            1, 
            verbose=verbose, 
            scaling_factor=scaling_factor,
            run_time_depth_index = 0
        )
        
        
        
        
        n1 = configs[0]
        self.BottleRep_neck_0 = BottleRep(
            c_out, 
            c_out, 
            max_kernel_size, 
            1, 
            number_blocks = n1, 
            verbose = verbose,
            scaling_factor = scaling_factor,
            run_time_depth_index = 1,
            keep_same_run_time = False,
            skip_connection    = False
        )
            
        up_ratio      = configs[1]
        self.up = nn.ConvTranspose2d(c_out, c_out, 
                                     up_ratio * 2, stride=up_ratio, 
                                     padding=up_ratio // 2, output_padding=0,
                                     groups=c_out, bias=False) # 变大wh
        fill_up_weights(self.up)
        
        
        
        n2    = configs[2]

        self.BottleRep_neck_1 = BottleRep(
            c_out, 
            c_out, 
            max_kernel_size, 
            1, 
            number_blocks        = n2, 
            verbose              = verbose,
            scaling_factor       = scaling_factor,
            run_time_depth_index = 0,
            keep_same_run_time   = False,
            skip_connection      = False
        )
    
    def forward(self, x1,x2):
        # x1 has more downration 2 times than x2

        x2 = self.project(x2)

        x2 = self.BottleRep_neck_0(x2)

        x2 = self.up(x2)
        
        return self.BottleRep_neck_1(x1+x2)
        
    def reset_active_depth_and_conv(self):
        
        self.project.reset_active_net_param()
        self.BottleRep_neck_0.reset_active_depth_and_conv()
        self.BottleRep_neck_1.reset_active_depth_and_conv()
        for param in self.up.parameters():
            param.requires_grad = True

    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):

        assert current_run_time_depth>= self.run_time_depth_index
        self.current_run_time_depth = current_run_time_depth
        

        self.project.set_current_runtime_depth_and_kernel(         current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        self.BottleRep_neck_0.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        self.BottleRep_neck_1.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)

        for param in self.up.parameters():
            param.requires_grad = False
        if current_run_time_depth == self.run_time_depth_index:
            for param in self.up.parameters():
                param.requires_grad = True
        elif current_run_time_depth>self.run_time_depth_index:
            if random or all_layers:
                for param in self.up.parameters():
                    param.requires_grad = True

    def random_set_current_runtime_depth_and_kernel( self,     all_layers=False,    active_net=False,     active_all_sub_net=False):
        
        self.project.set_current_runtime_depth_and_kernel(0,    None,    True, all_layers, active_net, active_all_sub_net)
        
        assert self.BottleRep_neck_0.run_time_depth_index_list is not None
        if self.BottleRep_neck_0.run_time_depth_index > 0:
            random_run_depth = random.choice([0]+self.BottleRep_neck_0.run_time_depth_index_list) 
        else:
            random_run_depth = random.choice(self.BottleRep_neck_0.run_time_depth_index_list) 
            
        self.BottleRep_neck_0.set_current_runtime_depth_and_kernel(random_run_depth,    None,    True, all_layers, active_net, active_all_sub_net)

        assert self.BottleRep_neck_1.run_time_depth_index_list is not None
        if self.BottleRep_neck_1.run_time_depth_index > 0:
            random_run_depth = random.choice([0]+self.BottleRep_neck_1.run_time_depth_index_list) 
        else:
            random_run_depth = random.choice(self.BottleRep_neck_1.run_time_depth_index_list) 
            
        self.BottleRep_neck_1.set_current_runtime_depth_and_kernel(random_run_depth,    None,    True, all_layers, active_net, active_all_sub_net)




class Neck(nn.Module):
    def __init__(self,  channels, max_kernel_size,     configs, verbose=False, scaling_factor = 0.5):
        super().__init__()

        self.channels = channels # [32, 48, 64, 128, 256, 512]
        assert len(self.channels) == 4
        assert len(configs) == len(self.channels)-1+2


        

        config_list = configs[0]
        self.up_level_1 = []
        for i, cfg in enumerate(config_list):

            self.up_level_1.append(BottleRep_neck_IDAUp(
                c_in             =  self.channels[i+1], 
                c_out            =  self.channels[i],
                max_kernel_size  =  max_kernel_size,
                configs          =  cfg,
                verbose          =  verbose,
                scaling_factor   =  0.5,
                run_time_depth_index= 0
            ))
            
        self.up_level_1 = nn.ModuleList(self.up_level_1)
          
        config_list = configs[1]
        self.up_level_2 = []
        for i, cfg in enumerate(config_list):
            self.up_level_2.append(BottleRep_neck_IDAUp(
                c_in             = self.channels[i+1], 
                c_out            = self.channels[i],
                max_kernel_size  = max_kernel_size,
                configs          = cfg,
                verbose          =  verbose,
                scaling_factor   =  0.5,
                run_time_depth_index= 0
            ))
            
        self.up_level_2 = nn.ModuleList(self.up_level_2)
        
        config_list = configs[2]    
        self.up_level_3 = []
        for i, cfg in enumerate(config_list):
            self.up_level_3.append(BottleRep_neck_IDAUp(
                c_in             = self.channels[i+1], 
                c_out            = self.channels[i],
                max_kernel_size  = max_kernel_size,
                configs          = cfg,
                verbose          =  verbose,
                scaling_factor   =  0.5,
                run_time_depth_index= 0
            ))
            
        self.up_level_3 = nn.ModuleList(self.up_level_3)
        
        cfg = configs[3]
        self.up_level_4 = BottleRep_neck_IDAUp(
            c_in             = self.channels[1], 
            c_out            = self.channels[0],
            max_kernel_size  = max_kernel_size,
            configs          = cfg,
            verbose          =  verbose,
            scaling_factor   =  0.5,
            run_time_depth_index= 0
        )
        
        
        
        cfg = configs[4]
        self.up_level_5 = BottleRep_neck_IDAUp(
            c_in             =self.channels[2], 
            c_out            =self.channels[0],
            max_kernel_size  =max_kernel_size,
            configs          = cfg,
            verbose          =  verbose,
            scaling_factor   =  0.5,
            run_time_depth_index= 0
        )
        
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

    def reset_active_depth_and_conv(self):
        
        for layer in self.up_level_1:
            layer.reset_active_depth_and_conv()
        
        for layer in self.up_level_2:
            layer.reset_active_depth_and_conv()

        for layer in self.up_level_3:
            layer.reset_active_depth_and_conv()

        self.up_level_4.reset_active_depth_and_conv()
        self.up_level_5.reset_active_depth_and_conv()

    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):

        for layer in self.up_level_1:
            layer.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        
        for layer in self.up_level_2:
            layer.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)

        for layer in self.up_level_3:
            layer.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        self.up_level_4.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        self.up_level_5.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)

    def random_set_current_runtime_depth_and_kernel( self,     all_layers=False,    active_net=False,     active_all_sub_net=False):
        
        for layer in self.up_level_1:
            layer.random_set_current_runtime_depth_and_kernel(all_layers, active_net, active_all_sub_net)
        
        for layer in self.up_level_2:
            layer.random_set_current_runtime_depth_and_kernel(all_layers, active_net, active_all_sub_net)

        for layer in self.up_level_3:
            layer.random_set_current_runtime_depth_and_kernel(all_layers, active_net, active_all_sub_net)
            
        self.up_level_4.random_set_current_runtime_depth_and_kernel(all_layers, active_net, active_all_sub_net)
        self.up_level_5.random_set_current_runtime_depth_and_kernel(all_layers, active_net, active_all_sub_net)





def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class OneForAll_SuperNetNetwork(nn.Module):
    def __init__(self,  network_config,  network_seed = 1, opt = None):
        super(OneForAll_SuperNetNetwork, self).__init__()
        
        self.opt = opt

        assert network_config["backbone"] is not None
        assert network_config["neck"] is not None
        assert network_config["head"] is not None
        
        backbone_config = network_config["backbone"]
        self.backbone = Backbone(
            config             = backbone_config,
            max_kernel_size    = opt.max_kernel_size,
            width_mult         = opt.width_mult,
            round_nearest      = 8,
            pre_img            = opt.pre_img,
            combine_style      = opt.combine_style,
            verbose            = opt.verbose,
            scaling_factor     = opt.scaling_factor,
            skip_connection    = opt.skip_connection,
        )

        
        channels = self.backbone.channels[2:]
        neck_config = network_config["neck"]
        self.neck = Neck(
            
            channels         = channels,
            max_kernel_size  = opt.max_kernel_size, 
            configs          = neck_config, 
            verbose          = opt.verbose,
            scaling_factor   = opt.scaling_factor
        )

        
        last_channel         = channels[0]
        self.heads           = opt.heads
        head_kernel          = 3
        prior_bias           = -4.6
        head_configs         = network_config["head"]





        
        for head in self.heads: # {'hm': num_cls, 'reg': 2, 'wh': 2}
            classes = self.heads[head]
            number_blocks =  head_configs[head]
            if number_blocks>0:
                convs = BottleRep(
                    c_in                = last_channel,
                    c_out               = last_channel,
                    max_kernel_size     = opt.max_kernel_size,
                    stride              =  1,
                    number_blocks       = number_blocks,
                    verbose             = opt.verbose,
                    scaling_factor      = opt.scaling_factor,
                    run_time_depth_index= 0,
                    keep_same_run_time  =False,
                    skip_connection     =False,
                )
            else:
                convs = None

            
            out = nn.Conv2d(last_channel, classes, kernel_size=1, stride=1, padding=0, bias=True)
            
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

        y = self.backbone(x, pre_img, pre_hm)

        feats = self.neck(*y[2:])
        out = []
        
        if self.opt.model_output_list:
            z = []

            for head in sorted(self.heads):

                z_temp = self.__getattr__(head)(feats)

                z.append(z_temp)
            out.append(z)
        else:
            z = {}
            for head in sorted(self.heads):
                z_temp = self.__getattr__(head)(feats)

                z[head] = z_temp

            out.append(z)

        return out

    def set_current_runtime_depth_and_kernel(self, 
                                             current_run_time_depth  = None, 
                                             kernel_index            = None, 
                                             
                                             random                  = False, 
                                             all_layers              = False,
                                             
                                             active_net              = False, 
                                             active_all_sub_net      = False,
                                             
                                            ):

        self.backbone.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        self.neck.set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)
        for head in sorted(self.heads):
            self.__getattr__(head)[0].set_current_runtime_depth_and_kernel(current_run_time_depth,    kernel_index,    random, all_layers, active_net, active_all_sub_net)

    def random_set_current_runtime_depth_and_kernel( self,     all_layers=False,    active_net=False,     active_all_sub_net=False):
        self.backbone.random_set_current_runtime_depth_and_kernel(all_layers,    active_net,     active_all_sub_net)
        self.neck.random_set_current_runtime_depth_and_kernel(all_layers,    active_net,     active_all_sub_net)
        for head in sorted(self.heads):
            assert self.__getattr__(head)[0].run_time_depth_index_list is not None
            random_run_depth = random.choice(self.__getattr__(head)[0].run_time_depth_index_list) 
            self.__getattr__(head)[0].set_current_runtime_depth_and_kernel(random_run_depth,    None,    True, all_layers, active_net, active_all_sub_net)


    def reset_active_depth_and_conv(self):
        
        self.backbone.reset_active_depth_and_conv()
        self.neck.reset_active_depth_and_conv()
        for head in sorted(self.heads):
            self.__getattr__(head)[0].reset_active_depth_and_conv()

    def get_the_run_time_depth_range(self):

        max_run_depth = 0

        for module in self.backbone.modules():
            if hasattr(module, 'run_time_depth_index_list'):
                if module.run_time_depth_index_list is not None:
                    max_temp = module.run_time_depth_index_list[-1]
                    if max_temp>max_run_depth:
                        max_run_depth = max_temp
            if hasattr(module, 'run_time_depth_index'):
                max_temp = module.run_time_depth_index
                if max_temp is not None and max_temp>max_run_depth:
                    max_run_depth = max_temp    

        for module in self.neck.modules():
            if hasattr(module, 'run_time_depth_index_list'):
                if module.run_time_depth_index_list is not None:
                    max_temp = module.run_time_depth_index_list[-1]
                    if max_temp>max_run_depth:
                        max_run_depth = max_temp
            if hasattr(module, 'run_time_depth_index'):
                max_temp = module.run_time_depth_index
                if max_temp is not None and max_temp>max_run_depth:
                    max_run_depth = max_temp    

        for head in sorted(self.heads):
            for module in self.__getattr__(head).modules():
                if hasattr(module, 'run_time_depth_index_list'):
                    if module.run_time_depth_index_list is not None:
                        max_temp = module.run_time_depth_index_list[-1]
                        if max_temp>max_run_depth:
                            max_run_depth = max_temp
                if hasattr(module, 'run_time_depth_index'):
                    max_temp = module.run_time_depth_index
                    if max_temp is not None and max_temp>max_run_depth:
                        max_run_depth = max_temp   
        return max_run_depth





