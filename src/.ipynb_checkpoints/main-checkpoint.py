#!/usr/bin/env python

#SBATCH --job-name=WACV23

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-user=zhou@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import os
import sys

sys.path.append(os.getcwd())
#sys.path.append('/pfs/data5/home/kit/tm/px6680/Conference/WACV/CenterTrack/src/')



# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import _init_paths
import os
# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization import tensor_quant
# from pytorch_quantization import calib
# from pytorch_quantization.tensor_quant import QuantDescriptor

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from model.networks.super_network import *
from torch.quantization import prepare_qat, get_default_qat_qconfig, convert
def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

  Dataset = get_dataset(opt.dataset)  # from .datasets.coco import COCO
  # TODO modify  default_resolution
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)


  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  if opt.arch == "oneforall" and opt.oneforall_progressive:
      max_depth_stage   = model.get_the_run_time_depth_range()
      max_kernel_stage  = len([0] + list(range(1,opt.max_kernel_size+1,2)))
      trainig_configs   = {}
      epoch_index       = 1
      for run_depth in range(max_depth_stage+1):
          for kernel_index in range(max_kernel_stage):
              for _ in range(opt.oneforall_pregressive_epoch):
                  trainig_configs[epoch_index] = ["progressive", run_depth, kernel_index]
                  epoch_index = epoch_index + 1
  else:
      trainig_configs = None

    

  optimizer = get_optimizer(opt, model)
  start_epoch = 0
  if opt.quant is None and opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  # TODO check the validation data
  if opt.val_intervals < opt.num_epochs or opt.test:
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
      pin_memory=True)

    if opt.test:
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.save_dir)
      return


  print('Setting up train data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
      num_workers=opt.num_workers, pin_memory=True, drop_last=True
  )

  print('Starting training...')

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'

    log_dict_train, _ = trainer.train(epoch, train_loader, trainig_configs)


    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        if opt.eval_val:
          val_loader.dataset.run_eval(preds, opt.save_dir)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.save_point:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
