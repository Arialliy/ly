import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from dataset_IDSMT import *
from dataset_IRDST import *
import matplotlib.pyplot as plt
from metrics import *
from model.MTTU_Net import MTTU_Net as MTTU_Net
import model.Config as config
import os
import time
from collections import OrderedDict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch VIRSTD test")
parser.add_argument("--model_names", default=['MTTU_Net'], nargs='+',
                    help="model_name: 'MTTU_Net'")
parser.add_argument("--pth_dirs", default=['IDSMT/MTTU_Net.pth.tar'], nargs='+',  help="checkpoint dir")
parser.add_argument("--dataset_dir", default='your dataset path', type=str, help="test_dataset_dir")
parser.add_argument("--dataset_names", default=['IDSMT'], nargs='+',
                    help="dataset_name: 'IDSMT', 'IRDST_real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std
  
def test():
    device = 'cuda:0'
    test_set = TestSetLoader(opt.dataset_dir, opt.test_dataset_name, frame_num=1)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    config_vit = config.get_MTTU_Net_config()
    net = MTTU_Net(config_vit, mode='test', deepsuper=True).to(device)

    try:
        state_dict = torch.load(opt.pth_dir)
        new_state_dict = OrderedDict()

        for k, v in state_dict['state_dict'].items():
            name = k[6:]
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
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

        ### save img
        if opt.save_img == True:
            img_save = transforms.ToPILImage()((pred[0,0,:,:] > opt.threshold).float().cpu())
            if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + str(seq_id.item())):
                os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + str(seq_id.item()))
            img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + str(seq_id.item()) + '/' + str(frame_id.item()) + '.png')


    results1 = eval_IoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, IoU:\t" + str(results1))
    print("PD(R), P, F1, FA:\t" + str(results2))
    opt.f.write("pixAcc, IoU:\t" + str(results1) + '\n')
    opt.f.write("PD(R), P, F1, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_100.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '/' + opt.model_name + '_100.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
        
