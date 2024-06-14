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

# Some functions of this script came from the repository of Uformer (https://github.com/ZhendongWang6/Uformer).

import torch
import os

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(name, checkpoint_dir, net, optimizer, epoch, iter):
    model_name = "epoch_{}_{}.pth".format(name, epoch)
    save_path = os.path.join(checkpoint_dir, model_name)
    torch.save({"epoch": epoch,
                "state_dict":net.state_dict(), 
                "optimizer" :optimizer.state_dict(),
                "iter" : iter}
                ,save_path)
    print('Checkpoint saved to {}'.format(save_path))        

def save_checkpoint_best(name, checkpoint_dir, net, optimizer, epoch, iter):
    model_name = "epoch_{}_best.pth".format(name)
    save_path = os.path.join(checkpoint_dir, model_name)
    torch.save({"epoch": epoch,
                "state_dict":net.state_dict(), 
                "optimizer" :optimizer.state_dict(),
                "iter" : iter}
                ,save_path)
    print('Checkpoint saved to {}'.format(save_path))      

def load_checkpoint(name, checkpoint_dir, net, epoch):
    model_name = "epoch_{}_{}.pth".format(name, epoch)
    load_path = os.path.join(checkpoint_dir, model_name)
    checkpoint = torch.load(load_path)
    net.load_state_dict(checkpoint["state_dict"])
    print('Load checkpoint on {}'.format(load_path))        

def load_start_epoch(name, checkpoint_dir,epoch):
    model_name = "epoch_{}_{}.pth".format(name, epoch)
    load_path = os.path.join(checkpoint_dir, model_name)
    checkpoint = torch.load(load_path)
    epoch = checkpoint["epoch"]
    return epoch

def load_start_iter(name, checkpoint_dir,epoch):
    model_name = "epoch_{}_{}.pth".format(name, epoch)
    load_path = os.path.join(checkpoint_dir, model_name)
    checkpoint = torch.load(load_path)
    iter = checkpoint["iter"]
    return iter

def load_optim(optimizer, name, checkpoint_dir,epoch):
    model_name = "epoch_{}_{}.pth".format(name, epoch)
    load_path = os.path.join(checkpoint_dir, model_name)
    checkpoint = torch.load(load_path)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def init_weights(net):
    if isinstance(net, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(net.weight)
        
        if net.bias is not None:
            net.bias.data.zero_()
