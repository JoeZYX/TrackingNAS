from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
import pickle

# python demo.py tracking --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3_night_combine_sub_cat/model_last.pth --arch custom --dataset coco_custom --nectwork_seed 109 --width_mult 0.3 --combine_style sub_cat --demo /pfs/data5/home/kit/tm/px6680/Conference/WACV/data/modified_vidoes --debug 4 --track_thresh 0.2

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
    print("-----------------------------------------------------------")
    video_list = os.listdir(opt.demo)
    video_list = [i for i in video_list if ".ipynb_checkpoints" not in i]
    print(video_list)
    for video in video_list:
      # if os.path.exists('/pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/sub_detrac/{}.pickle'.format(video)):
      #     print("skip-------------------- skip", video)
      #     continue
      video_path = os.path.join(opt.demo, video)
      os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
      opt.debug = max(opt.debug, 1)
      detector = Detector(opt)
    
      if video == 'webcam' or \
        video[video.rfind('.') + 1:].lower() in video_ext:
        is_video = True
        # demo on video stream
        cam = cv2.VideoCapture(0 if video == 'webcam' else video)
      else:
        is_video = False
        # Demo on images sequences
        if os.path.isdir(video_path):
          print("-----------++++++++++++++++++++++++++++++++++++++++++++++++++++++-------------")
          image_names = []
          ls = os.listdir(video_path)
          ls = [file for file in ls if "jpg" in file]
          ls.sort(key=lambda x: int(x.split(".")[0]))
          for file_name in ls:
              ext = file_name[file_name.rfind('.') + 1:].lower()
              if ext in image_ext:
                  image_names.append(os.path.join(video_path, file_name))
          print("image_names :" ,image_names)
          print("-------------++++++++++++++++++++++++++++++++++++++++++++++++++++++-----------")
        else:
          image_names = [video]
    
      # Initialize output video
      out = None
      out_name = video[video.rfind('/') + 1:]
      print('----------------------- out_name -----------------------', out_name, video)
      if opt.save_video:
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('../results/{}.mp4'.format(
          opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
            opt.video_w, opt.video_h))
      
      if opt.debug < 5:
        detector.pause = False
      cnt = 0
      results = {}
    
      while cnt<len(image_names):
          if is_video:
            _, img = cam.read()
            if img is None:
              save_and_exit(opt, out, results, out_name)
          else:
            if cnt < len(image_names):
              #print("load the {} img".format(cnt))
              img_name = image_names[cnt]
              #print(img_name)
              img = cv2.imread(image_names[cnt])
              img = cv2.resize(img, (640, 640))
            else:
              save_and_exit(opt, out, results, out_name)
          cnt += 1
    
          # resize the original video for saving video results
          if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))
    
          # skip the first X frames of the video
          if cnt < opt.skip_first:
            continue
          
          #cv2.imshow('input', img)
    
          # track or detect the image.
          ret = detector.run(img)
          #print(" detector run {} img done!".format(cnt))
          # log run time
          time_str = 'frame {} |'.format(cnt)
          for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
          #print(time_str)
    
          # results[cnt] is a list of dicts:
          #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
          #print(cnt, "   ",img_name,"  ", ret['results'])
          results[img_name] = ret['results']
    
          # save debug image to video
          if opt.save_video:
            out.write(ret['generic'])
            if not is_video:
              cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])
          if opt.save_img:
            cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])
          # esc to quit and finish saving video
          # if cv2.waitKey(1) == 27:
          #  save_and_exit(opt, out, results, out_name)
          #  return 
            
          # if not is_video and cnt == len(image_names):
          #   return 
      #save_and_exit(opt, out, results)
      with open('/pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/sub_detrac/{}.pickle'.format(video), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print("------------------------- saving ---------------------------")
      


def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
