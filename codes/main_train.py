#  Copyright (c) 2024 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import random
import numpy as np
import torch
import time
import datetime
import glob
from pytz import timezone


from config import config
from torch.utils.data import DataLoader
from utils.scheduler import GradualWarmupScheduler
import dataset
import utils.utils_image as util_image
import utils.utils_rend_img as util_rend
import utils.utils_options as option
import loss as L
import preprocess as pre
import eval
import model.model_joint_sa as model_joint_sa

def main():
    '''
    # =======================================
    # Step1 - Preparation
    # =======================================
    '''
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    # ----------------------------------------
    # Task
    # ----------------------------------------
    task_dir = os.path.join(config["data_dir"],config["task"])
    checkpoint_dir = os.path.join(task_dir, '__checkpoints__')         
    util_image.mkdir(checkpoint_dir)

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = config["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    str_print_train_seed = 'Random seed: {}\n'.format(seed)
    print(str_print_train_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # configure log
    # ----------------------------------------
    logfile_training = open(os.path.join(task_dir,'log_train_loss.txt'), 'a+')
    logfile_validation = open(os.path.join(task_dir, 'log_validate_loss.txt'), 'a+')

    logfile_training.write(str_print_train_seed)
    logfile_training.flush()

    '''
    # =======================================
    # Step2 - Dataloader
    # =======================================
    '''
    path_dataset = os.path.join(config["data_dir"], config["trainDatasetDirectory"])
    path_train_npz = os.path.join(path_dataset, str(config["train_input"]) + '_npz')
    file_list = []
    pathes = [fn for fn in glob.glob(os.path.join(path_train_npz +  '/*'))]
    for path in pathes:
        if path.endswith('.npz'):
            file_list.append(path)

    # if there are no npz file for train/test, convert img to npz
    if not (len(file_list) > 0):        
        print('check data set dir: {}'.format(path_train_npz))
        pre.construct_dataset_to_npz(config)

    # [setting] data loader
    train_set     = dataset.Dataset(config, train = True)
    train_loader  = DataLoader(dataset = train_set,
                                batch_size= config["batch_size"],
                                shuffle= True,
                                num_workers= config["dataloader_numworker"],
                                drop_last= True,
                                pin_memory= True)

    test_set           = dataset.Dataset(config, train = False)
    test_loader        = DataLoader(dataset = test_set, 
                                            batch_size= 1,
                                            num_workers= config["dataloader_numworker"])

    '''
    # =======================================
    # Step3 - Define Network
    # =======================================
    '''
    #  U-shaped joint self-attention & simple mlp
    net = model_joint_sa.JSA_transformer(img_size=config["patch_size"],
                                    embedded_dim=config["embed_dim"],
                                    win_size=8,
                                    projection_option='linear',
                                    ffn_option='mlp',
                                    depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
                                    in_x=config["x_dim"],
                                    in_f=config["f_dim"]
                                    )  

    # setting for multi gpu
    if config["multi_gpu"]:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    else:
        device = torch.device("cuda:0")
        net = net.to(device)


    '''
    # ----------------------------------------
    # Step4 Training
    # ----------------------------------------
    '''

    # [setting] optimizer & scheduler
    if config["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr_initial"], betas=(0.9, 0.999),eps=1e-8, weight_decay=config["weight_decay"])
    elif config["optimizer"] == 'adamw':
            optimizer = torch.optim.AdamW(net.parameters(), lr=config["lr_initial"], betas=(0.9, 0.999),eps=1e-8, weight_decay=config["weight_decay"])
    else:
        raise Exception("Error optimizer...")

    if config["wramup"]:
        print("Using warmup and cosine strategy!")
        warmup_epochs = config["warmup_epochs"]
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["max_epoch"]-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch = warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step() # warining
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step() # warining

   # [setting] retrain option
    if config["retrain"] == True:
        option.load_checkpoint(config["task"], checkpoint_dir, net, config["restore_epoch"])
        initial_epoch = option.load_start_epoch(config["task"], checkpoint_dir, config["restore_epoch"]) + 1 
        initial_iter = option.load_start_iter(config["task"], checkpoint_dir, config["restore_epoch"]) 
        lr = option.load_optim(optimizer, config["task"], checkpoint_dir,  config["restore_epoch"]) 

        for i in range(1, initial_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')
    else:
        initial_epoch = 1
        initial_iter = 0

    max_epoch = int(config["max_epoch"])
    current_iter = initial_iter



    print("Training Start")
    print('Start epoch: {} End epoch: {}'.format(initial_epoch,max_epoch))
    PSNR_average_best = 0

    for epoch in range(initial_epoch, max_epoch + 1):
        bat_cnt  = 0
        start_time = time.time()  
        net.train() 
        train_loss = 0
        epoch_best = 0

        for iteration, (input, gt, path) in enumerate(train_loader):
            # start time maker for one batch
            bat_st = time.time()         
                    
            bat_cnt += 1 
            current_iter += 1

            # aux_features = 0:2 - albedo  3 ch                           
                            # 3:5 - normal 3 ch
                            # 6 - depth    1 ch
            aux_features = input['aux']
            aux_features[:, :, :, 3:6] = torch.FloatTensor(util_rend.preprocess_normal(aux_features[:, :, :, 3:6]))  
            aux_features = aux_features.permute(0, 3, 1, 2)

            color_noisy = input["color"]
            color_noisy = util_rend.preprocess_specular(color_noisy)
            color_noisy = color_noisy.permute(0, 3, 1, 2)

            color_gt = gt['color']
            color_gt = util_rend.preprocess_specular(color_gt)
            color_gt = color_gt.permute(0, 3, 1, 2)            
            color_gt = color_gt.to("cuda")


            if config["multi_gpu"]:
                x = color_noisy.to("cuda")
                y = aux_features.to("cuda")
            else:
                x = color_noisy.to(device)
                y = aux_features.to(device)

            out = net(x=x, f=y)

            # zero the parameter gradients
            optimizer.zero_grad()
            denoised = out

            if config["multi_gpu"]:
                denoised = denoised.to("cuda")
                color_noisy = color_noisy.to("cuda")
            else:
                denoised = denoised.to(device)
                color_noisy = color_noisy.to(device)

            # [setting] loss
            # rel L2 loss
            loss = torch.mean(L.RelL2(denoised, color_gt))          

            # optimize parameter
            loss.backward()

            # update learning optimizer (lr)
            optimizer.step()

            # training infomation
            lr = option.get_lr(optimizer)
            bat_end = time.time() - bat_st
            str_print_info_per_batch = "[Epoch %03d / %03d] [Batch %04d / Iter %06d]  Loss: %0.8f,  lr: %0.7f,  Time: %0.4f" % (epoch, max_epoch, bat_cnt, current_iter, loss.item(), lr, bat_end) 
            print(str_print_info_per_batch)

            # accm training loss
            train_loss += loss.item()
        
        train_loss = train_loss / bat_cnt
        end_time = time.time() - start_time
        if config['timezone'] == None:
            str_data = datetime.datetime.now().astimezone(None).strftime("%Y-%m-%d %H:%M:%S")
        else:
            str_data = datetime.datetime.now().astimezone(timezone(config["timezone"])).strftime("%Y-%m-%d %H:%M:%S")

        str_print_train = "[Epoch %03d / %03d] [Iter %06d] (%f sec, %s) Avg Training loss : %0.8f\n" % (epoch, max_epoch, current_iter, end_time, str_data, train_loss)
        print(str_print_train)
        logfile_training.write(str_print_train)
        logfile_training.flush()

        # update scheduler
        scheduler.step()

        # Save Check point and Evaluate the method
        if epoch % config["snapshot"] == 0 or epoch == max_epoch:
            # save check point
            option.save_checkpoint(config["task"], checkpoint_dir, net, optimizer,epoch,current_iter)
            
            # evaluate network
            Loss_average, PSNR_average = eval.eval_train(net, test_loader, epoch)

            # log eval info of average value
            if config['timezone'] == None:
                str_data_eval = datetime.datetime.now().astimezone(None).strftime("%Y-%m-%d %H:%M:%S")
            else:
                str_data_eval = datetime.datetime.now().astimezone(timezone(config["timezone"])).strftime("%Y-%m-%d %H:%M:%S")

            str_print_evalutation = "[Epoch %03d / %03d] [Iter %06d] (%s) rel l2 Loss avg: %0.6f   PSNR avg: %0.6f\n" %(epoch, max_epoch, current_iter, str_data_eval, Loss_average, PSNR_average)
            print(str_print_evalutation)
            logfile_validation.write(str_print_evalutation)
            logfile_validation.flush()

            if PSNR_average > PSNR_average_best:
                PSNR_average_best = PSNR_average
                epoch_best = epoch
                iter_best = current_iter
                option.save_checkpoint_best(config["task"], checkpoint_dir, net, optimizer,epoch_best,iter_best)

if __name__ == '__main__':
    main()


