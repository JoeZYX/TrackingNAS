python main.py tracking --exp_id coco_custom_tracking_99_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 99 --width_mult 0.3  --num_epochs 1
python demo.py tracking --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_99_0_3/model_last.pth --arch custom --dataset coco_custom --nectwork_seed 99 --width_mult 0.3 --demo /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/imgs6 --debug 4 --track_thresh 0.2


python main.py tracking --exp_id coco_custom_tracking_100_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 100 --width_mult 0.3  --num_epochs 100
python demo.py tracking --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_100_0_3/model_last.pth --arch custom --dataset coco_custom --nectwork_seed 100 --width_mult 0.3 --demo /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/imgs7 --debug 4 --track_thresh 0.2



python main.py tracking --exp_id oneforalltest --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch oneforall --num_epochs 1

python demo.py tracking --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/oneforalltest/model_last.pth --arch oneforall --dataset coco_custom --demo /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/imgs7 --debug 4 --track_thresh 0.2




# # 105
# sbatch main.py tracking --exp_id coco_custom_tracking_105_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 105 --width_mult 0.3  --num_epochs 140
# #200
# sbatch main.py tracking --exp_id coco_custom_tracking_200_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 200 --width_mult 0.3  --num_epochs 140
# #300
# sbatch main.py tracking --exp_id coco_custom_tracking_300_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 300 --width_mult 0.3  --num_epochs 140
# # 400
# sbatch main.py tracking --exp_id coco_custom_tracking_400_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 400 --width_mult 0.3  --num_epochs 140
# # 600 
# sbatch main.py tracking --exp_id coco_custom_tracking_600_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 600 --width_mult 0.3  --num_epochs 140
# # 700
# sbatch main.py tracking --exp_id coco_custom_tracking_700_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 700 --width_mult 0.3  --num_epochs 140
# # 800
# sbatch main.py tracking --exp_id coco_custom_tracking_800_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 800 --width_mult 0.3  --num_epochs 140
# # 900
# sbatch main.py tracking --exp_id coco_custom_tracking_900_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 900 --width_mult 0.3  --num_epochs 140
# # 109
# sbatch main.py tracking --exp_id coco_custom_tracking_109_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3  --num_epochs 140
# # 110
# sbatch main.py tracking --exp_id coco_custom_tracking_110_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 110 --width_mult 0.3  --num_epochs 140
# #111
# sbatch main.py tracking --exp_id coco_custom_tracking_111_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 111 --width_mult 0.3  --num_epochs 140
# # 112
# sbatch main.py tracking --exp_id coco_custom_tracking_112_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 112 --width_mult 0.3  --num_epochs 140




# # 105
# sbatch main.py tracking --exp_id coco_custom_tracking_121_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 121 --width_mult 0.3  --num_epochs 140
# #200
# sbatch main.py tracking --exp_id coco_custom_tracking_122_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 122 --width_mult 0.3  --num_epochs 140
# #300
# sbatch main.py tracking --exp_id coco_custom_tracking_123_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 123 --width_mult 0.3  --num_epochs 140
# # 400
# sbatch main.py tracking --exp_id coco_custom_tracking_124_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 124 --width_mult 0.3  --num_epochs 140
# # 600 
# sbatch main.py tracking --exp_id coco_custom_tracking_125_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 125 --width_mult 0.3  --num_epochs 140
# # 700
# sbatch main.py tracking --exp_id coco_custom_tracking_127_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 127 --width_mult 0.3  --num_epochs 140
# # 800
# sbatch main.py tracking --exp_id coco_custom_tracking_1_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 1 --width_mult 0.3  --num_epochs 140
# # 900
# sbatch main.py tracking --exp_id coco_custom_tracking_2_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 2 --width_mult 0.3  --num_epochs 140
# # 109
# sbatch main.py tracking --exp_id coco_custom_tracking_3_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 3 --width_mult 0.3  --num_epochs 140
# # 110
# sbatch main.py tracking --exp_id coco_custom_tracking_99_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 99 --width_mult 0.3  --num_epochs 140
# #111
# sbatch main.py tracking --exp_id coco_custom_tracking_66_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 66 --width_mult 0.3  --num_epochs 140
# # 112
# sbatch main.py tracking --exp_id coco_custom_tracking_77_0_3 --tracking --dataset coco_custom --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 77 --width_mult 0.3  --num_epochs 140







# # 601
# sbatch main.py tracking --exp_id coco_custom_tracking_601_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 601 --width_mult 0.27  --num_epochs 140
# # 602
# sbatch main.py tracking --exp_id coco_custom_tracking_602_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 602 --width_mult 0.27  --num_epochs 140
# # 603
# sbatch main.py tracking --exp_id coco_custom_tracking_603_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 603 --width_mult 0.27  --num_epochs 140
# # 604
# sbatch main.py tracking --exp_id coco_custom_tracking_604_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 604 --width_mult 0.27  --num_epochs 140



# # 701
# sbatch main.py tracking --exp_id coco_custom_tracking_701_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 701 --width_mult 0.27  --num_epochs 140
# # 702
# sbatch main.py tracking --exp_id coco_custom_tracking_702_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 702 --width_mult 0.27  --num_epochs 140
# # 703
# sbatch main.py tracking --exp_id coco_custom_tracking_703_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 703 --width_mult 0.27  --num_epochs 140
# # 704
# sbatch main.py tracking --exp_id coco_custom_tracking_704_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 704 --width_mult 0.27  --num_epochs 140
# # 705
# sbatch main.py tracking --exp_id coco_custom_tracking_705_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 705 --width_mult 0.27  --num_epochs 140
# # 706
# sbatch main.py tracking --exp_id coco_custom_tracking_706_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 706 --width_mult 0.27  --num_epochs 140
# # 707
# sbatch main.py tracking --exp_id coco_custom_tracking_707_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 707 --width_mult 0.27  --num_epochs 140
# # 708
# sbatch main.py tracking --exp_id coco_custom_tracking_708_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 708 --width_mult 0.27  --num_epochs 140
# # 709
# sbatch main.py tracking --exp_id coco_custom_tracking_709_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 709 --width_mult 0.27  --num_epochs 140
# # 710
# sbatch main.py tracking --exp_id coco_custom_tracking_710_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 710 --width_mult 0.27  --num_epochs 140


# # 222
# sbatch main.py tracking --exp_id coco_custom_tracking_knn_222_0_27 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 222 --width_mult 0.27  --num_epochs 140

# # 1111
# sbatch main.py tracking --exp_id coco_custom_tracking_knn_1111_0_3 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 1111 --width_mult 0.3  --num_epochs 140

# # 1112
# sbatch main.py tracking --exp_id coco_custom_tracking_knn_1112_0_3 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 1112 --width_mult 0.3  --num_epochs 140

# # 1113
# sbatch main.py tracking --exp_id coco_custom_tracking_knn_1113_0_26 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 1113 --width_mult 0.26  --num_epochs 140

# # 1114
# sbatch main.py tracking --exp_id coco_custom_tracking_knn_1114_0_26 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 1114 --width_mult 0.26  --num_epochs 140

# # 1115
# sbatch main.py tracking --exp_id coco_custom_tracking_knn_1115_0_3 --tracking --dataset coco_custom --lr_step 15,25,35,45 --flip 0.0 --gpus 0 --batch_size 64 --lr 5e-4 --same_aug --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 1115 --width_mult 0.3  --num_epochs 140


python main.py tracking --exp_id tet --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_59.pth --quant_fuse


sbatch main.py tracking --exp_id quant_torch_109_fuse_input_out_quant --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_59.pth --quant_fuse

sbatch main.py tracking --exp_id quant_torch_109_no_fuse_input_out_quant --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_59.pth 



sbatch main.py tracking --exp_id quant_torch_109_fuse --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_59.pth --quant_fuse

sbatch main.py tracking --exp_id quant_torch_109_no_fuse --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_59.pth 


sbatch main.py tracking --exp_id quant_quant_109 --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant quant --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_59.pth




python main.py tracking --exp_id quant_test_109 --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant quant --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_59.pth


sbatch main.py tracking --exp_id coco_custom_tracking_109_0_3 --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --lr_step 10,25,40 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_tracking_109_0_3/model_last.pth --num_epochs 140


sbatch main.py tracking --exp_id coco_custom_109 --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --lr_step 25,40 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --resume --load_model /pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/exp/tracking/coco_custom_109/model_last.pth --num_epochs 140


python main.py tracking --exp_id small_quant_torch_109_no_fuse_input_out_quant --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch

sbatch main.py tracking --exp_id offsmall_quant_torch_109_no_fuse_input_out_quant --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch





sbatch main.py tracking --exp_id off_small_quant_torch_109_no_fuse_two_input_out_quant --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch

sbatch main.py tracking --exp_id off_small_quant_quant_109 --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant quant 



sbatch main.py tracking --exp_id off_small_quant_quant_109_quant_add --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant quant 


python main.py tracking --exp_id off_small_quant_torch_109_no_fuse_two_input_out_quant_cat_quant --tracking --dataset coco_custom --gpus 0 --batch_size 64 --lr 5e-4 --flip 0.0 --num_workers 16 --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --arch custom --nectwork_seed 109 --width_mult 0.3 --num_epochs 140 --quant torch


















