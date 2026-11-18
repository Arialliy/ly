import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *
import model.Config as config
from torch.utils.tensorboard import SummaryWriter
from model.MTTU_Net import MTTU_Net as MTTU_Net
from dataset_IDSMT import *
# from dataset_IRDST import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch VIRSTD train")
parser.add_argument("--model_names", default=['MTTU_Net'], nargs='+',  # SCTransNet
                    help="model_name: 'MTTU_Net'")
parser.add_argument("--dataset_names", default=['IDSMT'], nargs='+',
                    help="dataset_name: 'IDSMT', 'IRDST_real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='your dataset path', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--numFrame", type=int, default=5, help="Training frames")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, nargs='+',
                    help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--pretrained", default=None, nargs='+', help="Load pretrained checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict,
                    help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--intervals", type=int, default=1, help="Intervals for print loss")  # 10
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
    opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)

config_mttu_net = config.get_MTTU_Net_config()

device = 'cuda:0'

def train():
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, frame_num=opt.numFrame)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    net = Net(model_name=opt.model_name, mode='train').to(device)
    net.apply(weights_init_kaiming)
    net.train()

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
    if opt.pretrained:
        for pretrained_pth in opt.pretrained:
            if opt.dataset_name in pretrained_pth and opt.model_name in pretrained_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])

    ### Default settings of MTTU_Net
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.nEpochs, 'eta_min': 1e-5, 'last_epoch': -1}


    opt.nEpochs = opt.scheduler_settings['epochs']

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (imgs, gt_masks) in enumerate(train_loader):
            imgs, gt_masks = Variable(imgs).to(device), Variable(gt_masks).to(device)
            if imgs.shape[0] == 1:
                continue
            B, N, C, H, W = imgs.shape
            cur_memory_list = None
            loss_list = []
            for t in range(N):
                img = imgs[:, t, :, :, :]
                gt_mask = gt_masks[:, t, :, :, :]
                pred, cur_memory_list = net.forward(img, cur_memory_list)
                loss_t = net.loss(pred, gt_mask)
                loss_list.append(loss_t)

            loss = sum(loss_list) / N
            total_loss_epoch.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if (idx_epoch + 1) % opt.intervals == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,'
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []

        if (idx_epoch + 1) % 5 == 0:  # 50
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)

        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 5 != 0:  # 50
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)


def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, frame_num=1)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').to(device)
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()

    eval_IoU = IoU()
    eval_PD_FA = PD_FA()
    cur_memory_list = None
    for idx_iter, (img, gt_mask, size, seq_id, frame_id) in enumerate(test_loader):
        if frame_id.item() == 0:
            cur_memory_list = None
        img = Variable(img).to(device)
        with torch.no_grad():
            pred, cur_memory_list = net.forward(img[:,0,:,:,:], cur_memory_list)
        pred = pred[:, :, :size[1], :size[2]]
        gt_mask = gt_mask[:, 0, :, :size[1], :size[2]]
        eval_IoU.update((pred > opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size[1:])

    results1 = eval_IoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, IoU:\t" + str(results1))
    print("PD(R), P, F1, FA:\t" + str(results2))
    opt.f.write("pixAcc, IoU:\t" + str(results1) + '\n')
    opt.f.write("PD(R), P, F1, FA:\t" + str(results2) + '\n')


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # ************************************************loss*************************************************#
        self.cal_loss = nn.BCELoss(reduction='mean')

        if model_name == 'MTTU_Net':
            if mode == 'train':
                self.model = MTTU_Net(config_mttu_net, mode='train', deepsuper=True)
            else:
                self.model = MTTU_Net(config_mttu_net, mode='test', deepsuper=True)

    def forward(self, img, cur_memory_list):
        return self.model(img, cur_memory_list)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
