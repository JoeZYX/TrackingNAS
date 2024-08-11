from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import math
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
# ------------------------
# opts 文件改动了 文件路径
# ------------------------

mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
_data_rng = np.random.RandomState(123)

_eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
_eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)


_valid_ids = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
  24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
  37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
  58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
  72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
  82, 84, 85, 86, 87, 88, 89, 90]
cat_ids = {v: i + 1 for i, v in enumerate(_valid_ids)}


torch.manual_seed(opt.seed)
torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

Dataset = get_dataset(opt.dataset)  # from .datasets.coco import COCO
# TODO modify  default_resolution
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
#print("data_dir :", opt.data_dir)







split    = "val"
data_dir = os.path.join(opt.data_dir, 'coco')
img_dir  = os.path.join(data_dir, '{}2017'.format(split))
#print("img_dir : ",img_dir)

ann_path = os.path.join(
          data_dir, 'annotations', 
          'instances_{}2017.json').format(split)

#print("ann_path : ",ann_path)
#print("opt.flip : ",opt.flip) 0.5
#print("opt.max_frame_dist : ",opt.max_frame_dist)

# opt, split, ann_path, img_dir

if ann_path is not None and img_dir is not None:
    #print('==> initializing {} data from {}, \n images from {} ...'.format( split, ann_path, img_dir))
    coco_ = coco.COCO(ann_path)
    images = coco_.getImgIds()

    # print("first image :", coco_.dataset['images'][0])
    #  # 'license': 4, 
    #  #'file_name': '000000397133.jpg', '
    #  # coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 
    #  #'height': 427, 'width': 640, 
    #  # 'date_captured': '2013-11-14 17:02:52', 
    #  # 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
    
    
    #print("first image annotation:", coco_.dataset['annotations'][0])
    #  # segmentation  area  iscrowd  image_id  bbox  category_id id 

    if opt.tracking:
        # 因为coco数据库是没有视频信息的
        # 所以 这里给与每一个 vidow id ，这是不同的
        # 但是只有一张图片 所以 frameid是1
        # print("coco.dataset : ",coco_.dataset.keys()) #['info', 'licenses', 'images', 'annotations', 'categories']

        if not ('videos' in coco_.dataset):
            coco_.dataset['videos'] = []
            for i in range(len(coco_.dataset['images'])):
                img_id = coco_.dataset['images'][i]['id']
                coco_.dataset['images'][i]['video_id'] = img_id
                coco_.dataset['images'][i]['frame_id'] = 1
                coco_.dataset['videos'].append({'id': img_id})
    
        if not ('annotations' in coco_.dataset):
            print("there is no annotations")
        else:
            for i in range(len(coco_.dataset['annotations'])):
                coco_.dataset['annotations'][i]['track_id'] = i + 1


        # print("first image :", coco_.dataset['images'][0])
        # print("first image annotation:", coco_.dataset['annotations'][0])
        
        # print('Creating video index!')
        # 这里是要把 同一个video id的图片放到一个list中，dict-->list
        video_to_images = defaultdict(list)
        for image in coco_.dataset['images']:
            video_to_images[image['video_id']].append(image)

            

# -----------------------------
# 尝试load 一个 图片
# img, anns, img_info, img_path = self._load_data(index)
index  = 1
img_id = images[index]

# load the image
def load_image_anns(img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

img, anns, img_info, img_path = load_image_anns(img_id, coco_, img_dir)

#print("--------------------------")
#print(anns)
#print("--------------------------")
# get affine transformation parameter
height, width = img.shape[0], img.shape[1]

c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
s = max(img.shape[0], img.shape[1]) * 1.0 if not opt.not_max_crop \
  else np.array([img.shape[1], img.shape[0]], np.float32)
aug_s, rot, flipped = 1, 0, 0
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
trans_input = get_affine_transform( c, s, rot, [opt.input_w, opt.input_h])
trans_output = get_affine_transform(  c, s, rot, [opt.output_w, opt.output_h])

# 这里要正则化输入，调整channel，并且根据trans_input 改变图片
def _get_input( img, trans_input):
    inp = cv2.warpAffine(img, trans_input, 
                        (opt.input_w, opt.input_h),
                        flags=cv2.INTER_LINEAR)
    
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train' and not opt.no_color_aug:
        color_aug(data_rng, inp, _eig_val, _eig_vec)
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)
    return inp
inp = _get_input(img, trans_input)
ret = {'image': inp}
gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}


# load 前一张图片

def _load_pre_data(video_id, frame_id, sensor_id=1):
    img_infos = video_to_images[video_id]
    # If training, random sample nearby frames as the "previoud" frame
    # If testing, get the exact prevous frame
    if 'train' in split:
        # 筛选出适合范围内的图片  训练的时候 模拟一些变动
        img_ids = [(img_info['id'], img_info['frame_id']) for img_info in img_infos \
                   if abs(img_info['frame_id'] - frame_id) < opt.max_frame_dist and \
                   (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    else:
        
        # 由于是val或者test模式，所以一定要是前一帧的图片
        img_ids = [(img_info['id'], img_info['frame_id']) for img_info in img_infos \
                   if (img_info['frame_id'] - frame_id) == -1 and \
                   (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        if len(img_ids) == 0:
            # 如果 没有找到合适的图片，就选用当前帧作为前面的图片
            img_ids = [(img_info['id'], img_info['frame_id']) \
                       for img_info in img_infos \
                       if (img_info['frame_id'] - frame_id) == 0 and \
                       (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    
    rand_id = np.random.choice(len(img_ids))
    img_id, pre_frame_id = img_ids[rand_id]
    frame_dist = abs(frame_id - pre_frame_id)
    img, anns, _, _ = load_image_anns(img_id, coco_, img_dir)
    return img, anns, frame_dist

def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

def _get_bbox_output( bbox, trans_output, height, width):
    bbox = _coco_box_to_bbox(bbox).copy()

    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    for t in range(4):
        rect[t] =  affine_transform(rect[t], trans_output)
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

    bbox_amodal = copy.deepcopy(bbox)
    
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, opt.output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, opt.output_h - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    return bbox, bbox_amodal

def _get_pre_dets( anns, trans_input, trans_output):
    hm_h, hm_w = opt.input_h, opt.input_w
    assert hm_h==640
    assert hm_w==640
    down_ratio = opt.down_ratio
    assert down_ratio==4
    trans = trans_input
    reutrn_hm = opt.pre_hm
    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
    pre_cts, track_ids = [], []
    for ann in anns:
        if ann['category_id'] not in cat_ids.keys():
            continue
        cls_id = int(cat_ids[ann['category_id']])
        if cls_id > opt.num_classes or cls_id <= -99 or ('iscrowd' in ann and ann['iscrowd'] > 0):
            continue
        bbox = _coco_box_to_bbox(ann['bbox'])
        bbox[:2] = affine_transform(bbox[:2], trans)
        bbox[2:] = affine_transform(bbox[2:], trans)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        max_rad = 1
        if (h > 0 and w > 0):
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius)) 
            max_rad = max(max_rad, radius)
            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            #print("previous ct :" ,ct/ down_ratio)
            ct0 = ct.copy()
            conf = 1

            ct[0] = ct[0] + np.random.randn() * opt.hm_disturb * w
            ct[1] = ct[1] + np.random.randn() * opt.hm_disturb * h
            conf = 1 if np.random.random() > opt.lost_disturb else 0
        
            ct_int = ct.astype(np.int32)
            if conf == 0:
                pre_cts.append(ct / down_ratio)
            else:
                pre_cts.append(ct0 / down_ratio)

            track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
            if reutrn_hm:
                draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

            if np.random.random() < opt.fp_disturb and reutrn_hm:
                ct2 = ct0.copy()
                # Hard code heatmap disturb ratio, haven't tried other numbers.
                ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
                ct2_int = ct2.astype(np.int32)
                draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

    return pre_hm, pre_cts, track_ids


if opt.tracking:
    pre_image, pre_anns, frame_dist = _load_pre_data( 
        img_info['video_id'],
        img_info['frame_id'], 
        img_info['sensor_id'] if 'sensor_id' in img_info else 1)
    print("opt.same_aug_pre",opt.same_aug_pre)
    print("frame_dist",frame_dist)

    trans_input_pre = trans_input 
    trans_output_pre = trans_output

    pre_img = _get_input(pre_image, trans_input_pre)
    
    pre_hm, pre_cts, track_ids = _get_pre_dets( pre_anns, trans_input_pre, trans_output_pre)
    ret['pre_img'] = pre_img

    
# -------------------- init Ret -----------
# 到这个位置的时候 'image', 'pre_img' ， ret里面只有这些东西
max_objs = 128

max_objs = max_objs * opt.dense_reg # 1

ret['hm'] = np.zeros(
  (opt.num_classes, opt.output_h, opt.output_w),  # 80, 128, 128
  np.float32)
ret['ind'] = np.zeros((max_objs), dtype=np.int64)
ret['cat'] = np.zeros((max_objs), dtype=np.int64)
ret['mask'] = np.zeros((max_objs), dtype=np.float32)
num_joints = 17
regression_head_dims = {
  'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4, 
  'nuscenes_att': 8, 'velocity': 3, 'hps': num_joints * 2, 
  'dep': 1, 'dim': 3, 'amodel_offset': 2}
print("opt.heads",opt.heads)
# hm:6 reg:2 wh:2 tracking:2
for head in regression_head_dims:
    if head in opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []
        

# ret
# hm  number of class x   input_w/donw_ratio  x  input_h/down_ration
# ind
# cat
# mask
# reg  max_objs x 2
# reg_mask max_objs x 2
# wh  max_objs x 2
# wh_mask   max_objs x 2
# tracking  max_objs x 2
# tracking_mask max_objs x 2
        
# gt_det  # 'bboxes': [], 'scores': [], 'clses': [], 'cts': [], 'reg': [], 'wh': [], 'tracking': []
# for key in ret.keys():
#     print(key,ret[key].shape)

rest_focal_length = 1200
def _get_calib( img_info, width, height):
    if 'calib' in img_info:
        calib = np.array(img_info['calib'], dtype=np.float32)
    else:
        calib = np.array([[rest_focal_length, 0, width / 2, 0], 
                          [0, rest_focal_length, height / 2, 0], 
                          [0, 0, 1, 0]])
    return calib
calib = _get_calib(img_info, width, height)


num_objs = min(len(anns), max_objs)









for k in range(num_objs):
    ann = anns[k]
    cls_id = int(cat_ids[ann['category_id']])
    if cls_id > opt.num_classes or cls_id <= -999:
        continue
    bbox, bbox_amodal = _get_bbox_output( ann['bbox'], trans_output, height, width)
    # bbox 是 clip 之后的
    # bbox_amodal 是没有经过clip的
    # ------------------ _add_instance ------------------
    # 输入有： ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, calib, pre_cts, track_ids
    # gt_det  # 'bboxes': [], 'scores': [], 'clses': [], 'cts': [], 'reg': [], 'wh': [], 'tracking': []
    # k  第几个annotation  for 循环中的
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if h <= 0 or w <= 0:
        continue
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius)) 
    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    #print("current ct :",ct)
    ct_int = ct.astype(np.int32)
    ret['cat'][k] = cls_id - 1  # 维度就是 num obj max 有一个 添一个
    ret['mask'][k] = 1  # 对应位置 mask  添为1  表示有
    if 'wh' in ret:
        ret['wh'][k] = 1. * w, 1. * h
        ret['wh_mask'][k] = 1
    # ind 表示 绝对位置
    ret['ind'][k] = ct_int[1] * opt.output_w + ct_int[0]
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)
    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    gt_det['scores'].append(1)
    gt_det['clses'].append(cls_id - 1)
    gt_det['cts'].append(ct)
    
    
    if 'tracking' in opt.heads:
        if ann['track_id'] in track_ids:
            pre_ct = pre_cts[track_ids.index(ann['track_id'])]
            ret['tracking_mask'][k] = 1
            ret['tracking'][k] = pre_ct - ct_int
            #print("--------------",ret['tracking'][k])
            gt_det['tracking'].append(ret['tracking'][k])
        else:
            gt_det['tracking'].append(np.zeros(2, np.float32))




print("opt.num_stacks : ", opt.num_stacks)