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

import torch
import os
import glob
import random
import numpy as np
import torch.utils.data as torch_dataset

import utils.utils_image as util_image
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_image_name(directory, endMatch):
    file_list = []
    pathes = [fn for fn in glob.glob(os.path.join(directory +  '/*'))]
    for path in pathes:
        if path.endswith(endMatch):
            file_list.append(path)

    file_list = natural_sort(file_list)
    return file_list


def check_paris(color_filenames, ref_filenames):
    for (c, r) in zip(color_filenames, ref_filenames):
        cname = os.path.basename(c).split('.')[0].split('_')
        c_chk = cname[0]
        r_chk = os.path.basename(r).split('.')[0]
        check_token_1 = False
        if c_chk != r_chk:
            # print('Wrong pair!', c_chk, r_chk)
            check_token_1 = True

        cname2 = os.path.basename(c).split('.')[0]
        c_chk2 = cname2
        r_chk2 = os.path.basename(r).split('.')[0]
        check_token_2 = False
        if c_chk2 != r_chk2:
            # print('Wrong pair!', c_chk, r_chk)
            check_token_2 = True
        
        check_token_tot = (check_token_1) and (check_token_2)
        
        assert not check_token_tot,  ('Wrong pair!', c, r)

def check_paris_test(color_filenames, ref_filenames):
    for (c, r) in zip(color_filenames, ref_filenames):
        cname = os.path.basename(c).split('.')[0].split('_')
        c_chk = cname[0]
        r_chk = os.path.basename(r).split('.')[0]
        check_token_1 = False
        if c_chk != r_chk:
            # print('Wrong pair!', c_chk, r_chk)
            check_token_1 = True

        cname2 = os.path.basename(c).split('.')[0]
        c_chk2 = cname2
        r_chk2 = os.path.basename(r).split('.')[0]
        check_token_2 = False
        if c_chk2 != r_chk2:
            # print('Wrong pair!', c_chk, r_chk)
            check_token_2 = True
        
        check_token_tot = (check_token_1) and (check_token_2)
        
        assert not check_token_tot,  ('Wrong pair!', c, r)

class Dataset(torch_dataset.Dataset):
    def __init__(self, config, train = True):
        super(Dataset, self).__init__()
        # ---------------------------------
        # preparation
        # ---------------------------------
        self.config = config
        if train:
            path_train = os.path.join(config["trainDatasetDirectory"])
            self.data_input = os.path.join(path_train, str(config["train_input"]) + '_npz')
            self.data_target = os.path.join(path_train, str(config["train_target"]) + '_npz')
        elif not train:
            path_test  = os.path.join(config["testDatasetDirectory"])
            self.data_input = os.path.join(path_test, str(config["test_input"]) + '_npz')
            self.data_target  = os.path.join(path_test, str(config["test_target"]) + '_npz')

        self.input_list = load_image_name(self.data_input, '.npz')
        self.targetlist = load_image_name(self.data_target, '.npz')    

        check_paris(self.input_list, self.targetlist)        

        self.paths = np.stack([self.input_list, self.targetlist], axis = 1)
        numData = len(self.paths)

        if train:
            print('[%s] num train data: %d' % (config['trainDatasetDirectory'], numData))
        else:
            print('[%s] num test data: %d' % (config['testDatasetDirectory'], numData))

    
        self.patch_size = config["patch_size"] 
        self.patch_stride_size = config["patch_stride_size"] 
        self.patch_shuffle = config["dataloader_shuffle"] 
        self.train = train

    def __getitem__(self, index):
        # -------------------------------------
        # get train image
        # -------------------------------------
        input = np.load(self.paths[index][0])
        gt = np.load(self.paths[index][1])

        if self.train == True:
            """
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            """
            H, W = input['color'].shape[:2]
            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_input = {}
            patch_gt = {}
            # keys = ['color','albedo', 'normal','depth']
            keys = input.keys()
            keys_gt = gt.keys()
            for key in keys:
                patch_input[key] = input[key][rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            for key in keys_gt:
                patch_gt[key]    = gt[key][rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            if self.config["aug_mode"]:
                # ---------------------------------
                # augmentation - flip, rotate
                # ---------------------------------
                mode = random.randint(0, 7)
                for key in keys:
                    patch_input[key] = util_image.augment_img(patch_input[key], mode=mode)
                for key in keys_gt:
                    patch_gt[key] = util_image.augment_img(patch_gt[key], mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            for key in keys:
                patch_input[key] = util_image.uint2tensor3_not_normalize_not_permute(patch_input[key])
            for key in keys_gt:
                patch_gt[key] = util_image.uint2tensor3_not_normalize_not_permute(patch_gt[key])

            img_input, img_gt = patch_input, patch_gt


        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            keys = input.keys()
            keys_gt = gt.keys()
            img_input = {}
            img_gt = {}
            for key in keys:
                img_input[key] = input[key]
                img_input[key] = util_image.uint2tensor3_not_normalize_not_permute(img_input[key])

            for key in keys_gt:
                img_gt[key]    = gt[key]
                img_gt[key] = util_image.uint2tensor3_not_normalize_not_permute(img_gt[key])


        return img_input, img_gt, self.paths[index][0]

    def __len__(self):
        return len(self.paths)


            


                


