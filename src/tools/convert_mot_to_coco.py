import os
import numpy as np
import json
import cv2

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = '../../data/mot17/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['train_half', 'val_half', 'train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + (split if not HALF_VIDEO else 'train')
    out_path = OUT_PATH + '{}.json'.format(split)
    # out 包含 images annotations videos categories，！！！ categories要预先定义 id是什么 name是什么
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'pedestrain'}],
           'videos': []}
    seqs = os.listdir(data_path)
    # seqs包含了所有的视频
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      if '.DS_Store' in seq:
        continue
      if 'mot17' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
        continue
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'
      images = os.listdir(img_path)
      num_images = len([image for image in images if 'jpg' in image])

      image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        det_path = seq_path + 'det/det.txt'
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')



        print(' {} ann images'.format(int(anns[:, 0].max())))
        for i in range(anns.shape[0]):
          frame_id = int(anns[i][0])
          if (frame_id - 1 < image_range[0] or frame_id - 1> image_range[1]):
            continue
          track_id = int(anns[i][1])
          cat_id = int(anns[i][7])
          ann_cnt += 1
          if not ('15' in DATA_PATH):
            if not (float(anns[i][8]) >= 0.25):
              continue
            if not (int(anns[i][6]) == 1):
              continue
            if (int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]): # Non-person
              continue
            if (int(anns[i][7]) in [2, 7, 8, 12]): # Ignored person
              category_id = -1
            else:
              category_id = 1
          else:
            category_id = 1
          ann = {'id': ann_cnt,
                 'category_id': category_id,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': anns[i][2:6].tolist(),
                 'conf': float(anns[i][6])}
          out['annotations'].append(ann)
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
        
        