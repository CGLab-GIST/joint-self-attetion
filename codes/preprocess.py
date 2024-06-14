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

import pyexr
import numpy as np
import os
import utils.utils_image as util_image
import utils.utils_rend_img as util_rend
import glob
import multiprocessing
import parmap
from functools import partial

def load_image_name(directory, endMatch):
    file_list = []
    pathes = [fn for fn in glob.glob(os.path.join(directory +  '/*'))]
    for path in pathes:
        if path.endswith(endMatch):
            file_list.append(path)

    return file_list
    
def save_images_to_npz(filenames):
    print('cpu count:', multiprocessing.cpu_count())

    # make new folder for npy files
    file_dir = os.path.split(filenames[0]) 
    new_dir = file_dir[0] + '_npz'
    util_image.mkdir(new_dir)

    parmap.map(partial(convert_images_to_npz), filenames, pm_pbar=True, pm_processes=multiprocessing.cpu_count())


def save_test_images_to_npz(filenames):
    print('cpu count:', multiprocessing.cpu_count())

    # make new folder for npy files
    file_dir = os.path.split(filenames[0]) 
    new_dir = file_dir[0] + '_npz'
    util_image.mkdir(new_dir)

    parmap.map(partial(convert_test_images_to_npz), filenames, pm_pbar=True, pm_processes=multiprocessing.cpu_count())

def convert_test_images_to_npz(filename):
    file_basename = os.path.basename(filename).split('.')[0]
    file_name_split = file_basename.split('_') # 
    file_dir = os.path.split(filename) 
    file_type = os.path.splitext(filename) 

    new_dir = file_dir[0] + '_npz'
    new_filename = os.path.join(new_dir,os.path.basename(filename))
    new_filename = new_filename.replace('exr', 'npz')

    exr = pyexr.open(filename)    
    data_tmp = exr.get_all()

    types = ['default']
    names = ['color']
    data = {}
    for type, name in zip(types, names):
        data[name] = data_tmp[type]

    # nan to 0.0, inf to finite number
    for channel_name, channel_value in data.items():
        data[channel_name] = np.nan_to_num(channel_value)

    # clip data to avoid negative values
    data['color'] = np.clip(data['color'], 0, np.max(data['color']))
    np.savez_compressed(new_filename, color = data['color'])
    print('convert {} to {}'.format(filename, new_filename))

def convert_images_to_npz(filename):
    file_basename = os.path.basename(filename).split('.')[0]
    file_name_split = file_basename.split('_') # 
    file_dir = os.path.split(filename) 
    file_type = os.path.splitext(filename) 

    new_dir = file_dir[0] + '_npz'
    new_filename = os.path.join(new_dir,os.path.basename(filename))
    new_filename = new_filename.replace('exr', 'npz')

    exr = pyexr.open(filename)    
    data_tmp = exr.get_all()

    types = ['default','albedo', 'normal','depth']
    names = ['color','albedo', 'normal','depth']
    data = {}
    for type, name in zip(types, names):
        data[name] = data_tmp[type]

    # nan to 0.0, inf to finite number
    for channel_name, channel_value in data.items():
        data[channel_name] = np.nan_to_num(channel_value)

    # clip data to avoid negative values
    data['color'] = np.clip(data['color'], 0, np.max(data['color']))

    # normalize auxiliary features to [0.0, 1.0]
    data['depth'] = util_rend.preprocess_depth(data['depth'].copy())

    aux_features = np.concatenate((data['albedo'].copy(),
                                   data['normal'].copy(),
                                   data['depth'].copy()), axis=2)
    data['aux'] = aux_features

    np.savez_compressed(new_filename, color = data['color'], aux = data['aux'])
    print('convert {} to {}'.format(filename, new_filename))

def construct_dataset_to_npz(config):
    path_train = os.path.join(config["trainDatasetDirectory"])
    path_test  = os.path.join(config["testDatasetDirectory"])

    path_train_input = os.path.join(path_train, str(config["train_input"]))
    path_train_target = os.path.join(path_train, str(config["train_target"]))
    
    path_test_input = os.path.join(path_test, str(config["test_input"]))
    path_test_target = os.path.join(path_test, str(config["test_target"]))


    train_input_list = load_image_name(path_train_input, '.exr')
    save_images_to_npz(train_input_list)
    print('Done: construct train input')

    train_target_list = load_image_name(path_train_target, '.exr')
    save_test_images_to_npz(train_target_list)
    print('Done: construct train target')

    test_input_list = load_image_name(path_test_input, '.exr')
    save_images_to_npz(test_input_list)
    print('Done: construct test input')

    test_target_list = load_image_name(path_test_target, '.exr')
    save_test_images_to_npz(test_target_list)
    print('Done: construct test target')


def construct_test_dataset_to_npz(config):
    path_train = os.path.join(config["trainDatasetDirectory"])
    path_test  = os.path.join(config["testDatasetDirectory"])
    
    path_test_input = os.path.join(path_test, str(config["test_input"]))
    path_test_target = os.path.join(path_test, str(config["test_target"]))

    test_input_list = load_image_name(path_test_input, '.exr')
    save_images_to_npz(test_input_list)
    print('Done: construct test input')

    test_target_list = load_image_name(path_test_target, '.exr')
    save_test_images_to_npz(test_target_list)
    print('Done: construct test target')