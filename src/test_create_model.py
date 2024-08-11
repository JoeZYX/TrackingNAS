from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from collections import defaultdict
import pycocotools.coco as coco
import cv2
import copy
import numpy as np


opt = opts().parse()



print("---------------------- use {} Dataset ---------------------".format(opt.dataset))
Dataset = get_dataset(opt.dataset)
# 这里要更新 输入size  输出size  以及根据要求的任务增加模型的头数
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)



print('Creating model...')
model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

split = "val"

data_dir = os.path.join(opt.data_dir, 'coco')
img_dir = os.path.join(data_dir, '{}2017'.format(split))

print("--------------------- data_dir {} -------------------------".format(data_dir))
print("--------------------- img_dir  {} -------------------------".format(img_dir))
print("---------------------opt.trainval: {}----------------------".format(opt.trainval))
if opt.trainval:
    split = 'test'
    ann_path = os.path.join(
        data_dir, 'annotations', 
        'image_info_test-dev2017.json')
else:
    ann_path = os.path.join(
        data_dir, 'annotations', 
        'instances_{}2017.json').format(split)

print(ann_path)


print(opt)






# ------------------------
# opts 文件改动了 文件路径
# ------------------------
# python test_create_model.py tracking --exp_id coco_tracking --tracking --load_model ../models/ctdet_coco_dla_2x.pth  --gpus 0,1,2,3,4,5,6,7 --batch_size 128 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --dla_node conv

# python info.py tracking --exp_id coco_tracking --tracking --load_model ../models/ctdet_coco_dla_2x.pth  --gpus 0,1,2,3,4,5,6,7 --batch_size 128 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --dla_node conv

# mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
# std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
# _data_rng = np.random.RandomState(123)

# _eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
# _eig_vec = np.array([
#     [-0.58752847, -0.69563484, 0.41340352],
#     [-0.5832747, 0.00994535, -0.81221408],
#     [-0.56089297, 0.71832671, 0.41158938]
# ], dtype=np.float32)

# torch.manual_seed(opt.seed)
# torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

# Dataset = get_dataset(opt.dataset)  # from .datasets.coco import COCO
# # TODO modify  default_resolution
# opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
# #print("data_dir :", opt.data_dir)



# print("opt.num_classes : ", opt.num_classes)

# print("opt.input_h : ",opt.input_h)
# print("opt.input_w : ",opt.input_w)
# print("opt.down_ratio : ",opt.down_ratio)
# print("opt.output_h : ",opt.output_h)
# print("opt.output_w : ",opt.output_w)

# print("opt.input_res : ",opt.input_res)
# print("opt.output_res : ",opt.output_res)



# # ----------------  test creat model ---------------------
# print('Creating model...')
# arch = opt.arch

# num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
# arch       = arch[:arch.find('_')] if '_' in arch else arch
# from lib.model.networks.dla import DLASeg
# from lib.model.networks.resdcn import PoseResDCN
# from lib.model.networks.resnet import PoseResNet
# from lib.model.networks.dlav0 import DLASegv0
# from lib.model.networks.generic_network import GenericNetwork

# _network_factory = {
#   'resdcn': PoseResDCN,
#   'dla': DLASeg,
#   'res': PoseResNet,
#   'dlav0': DLASegv0,
#   'generic': GenericNetwork
# }

# model_class = _network_factory[arch]
# head = opt.heads
# head_conv = opt.head_conv

# print(arch)
# print(head)
# print(head_conv)

# print(opt.prior_bias)
# model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)

# #print(model)