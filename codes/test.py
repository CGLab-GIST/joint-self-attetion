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
import datetime
import glob
from pytz import timezone
from torch.utils.data import DataLoader

from config import config
import utils.utils_image as util_image
import utils.utils_options as option
import dataset
import model.model_joint_sa as model_joint_sa
import preprocess as pre
import eval as eval



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
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # configure log
    # ----------------------------------------
    logfile_validation = open(os.path.join(task_dir, 'log_test_loss.txt'), 'a+')

    '''
    # =======================================
    # Step2 - Dataloader
    # =======================================
    '''
    path_dataset = os.path.join(config["testDatasetDirectory"])
    path_train_npz = os.path.join(path_dataset, str(config["test_input"]) + '_npz')
    file_list = []
    pathes = [fn for fn in glob.glob(os.path.join(path_train_npz +  '/*'))]
    for path in pathes:
        if path.endswith('.npz'):
            file_list.append(path)

    # if there are no npz file for train/test, convert img to npz
    if not (len(file_list) > 0):        
        print('check data set dir: {}'.format(path_train_npz))
        pre.construct_test_dataset_to_npz(config)
        
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
                                    embedded_dim=32,
                                    win_size=8,
                                    projection_option='linear',
                                    ffn_option='mlp',
                                    depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
                                    in_x=3,
                                    in_f=7
                                    )  

    
    # use single GPU
    # setting for multi gpu
    if config["multi_gpu"]:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    else:
        device = torch.device("cuda:0")
        net = net.to(device)

    # load check point
    test_epoch = config["load_epoch"]
    option.load_checkpoint(config["task"], checkpoint_dir, net, test_epoch)
    
    '''
    # ----------------------------------------
    # Step4 Testing
    # ----------------------------------------
    '''
    print("Testing Start")
    # evaluate network
    Loss_average, PSNR_average = eval.eval_test(net, test_loader, test_epoch)

    # log eval info of average value
    if config['timezone'] == None:
        str_data_eval = datetime.datetime.now().astimezone(None).strftime("%Y-%m-%d %H:%M:%S")
    else:
        str_data_eval = datetime.datetime.now().astimezone(timezone(config["timezone"])).strftime("%Y-%m-%d %H:%M:%S")
    if type(test_epoch) is str: 
        str_print_evalutation = "[Epoch %s / %s] (%s) Loss avg: %0.6f   PSNR avg: %0.6f\n" %(test_epoch, test_epoch, str_data_eval, Loss_average, PSNR_average)
    elif type(test_epoch) is int:
        str_print_evalutation = "[Epoch %03d / %03d] (%s) Loss avg: %0.6f   PSNR avg: %0.6f\n" %(test_epoch, test_epoch, str_data_eval, Loss_average, PSNR_average)

    print(str_print_evalutation)
    logfile_validation.write(str_print_evalutation)
    logfile_validation.flush()
    print("Testing Done")



if __name__ == '__main__':
    main()

