import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
from torch.nn import init
from warmup_scheduler import GradualWarmupScheduler
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def seed_pytorch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        init.xavier_normal(m.weight.data)
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if classname.find('D_ConvLSTMCell') == -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
                
def random_crop(imgs, masks, patch_size, pos_prob=None):
    t, h, w = imgs.shape
    if min(h, w) < patch_size:
        imgs = np.pad(imgs, ((0,0),(0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        masks = np.pad(masks, ((0,0),(0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        h, w = imgs.shape

    cur_prob = random.random()
    if pos_prob == None or cur_prob > pos_prob or masks.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(masks > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0])-1)
        h_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], h-patch_size))
        w_start = random.randint(max(0, loc[2][idx] - patch_size), min(loc[2][idx], w-patch_size))
    
    h_end = h_start + patch_size
    w_end = w_start + patch_size
    imgs_patch = imgs[:, h_start:h_end, w_start:w_end]
    masks_patch = masks[:, h_start:h_end, w_start:w_end]
        
    return imgs_patch, masks_patch

def Normalized(img, img_norm_cfg):
    return (img-img_norm_cfg['mean'])/img_norm_cfg['std']
    
def Denormalization(img, img_norm_cfg):
    return img*img_norm_cfg['std']+img_norm_cfg['mean']



def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    if optimizer_name == 'Adamweight':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'], weight_decay=1e-3)

    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'], momentum=0.9, weight_decay=scheduler_settings['weight_decay'])
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'], gamma=scheduler_settings['gamma'])

    elif scheduler_name == 'CosineAnnealingLR':
        warmup_epochs = 10
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'] - warmup_epochs, eta_min=scheduler_settings['eta_min'])
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    return optimizer, scheduler

def PadImg(img, times=32):
    t, h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0,0),(0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0,0),(0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img        
