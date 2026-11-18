from utils import *
import matplotlib.pyplot as plt
import os
import json
import shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def Normalized(img):
    return (img-98.27898754143953)/45.15647234095911


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, frame_num):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name + '/train'
        self.patch_size = patch_size
        self.frame_num = frame_num

        self.seq_list = sorted([int(f.split('.')[0]) for f in os.listdir(os.path.join(self.dataset_dir, 'labels'))])
        self.frames_per_seq = 300

        self.tranform = augumentation()
        
    def __getitem__(self, idx):
        seq_idx = idx // self.frames_per_seq
        frame_idx = idx % self.frames_per_seq

        img_list = []
        mask_list = []

        for i in range(self.frame_num):
            img = Image.open(self.dataset_dir + '/images/' + '{}/'.format(self.seq_list[seq_idx]) + '{}.png'.format(max(frame_idx-i, 0))).convert('I')
            mask = Image.open(self.dataset_dir + '/masks/' + '{}/'.format(self.seq_list[seq_idx]) + '{}.png'.format(max(frame_idx-i, 0)))

            img = Normalized(np.array(img, dtype=np.float32))
            mask = np.array(mask, dtype=np.float32) / 255.0

            img_list.append(img)
            mask_list.append(mask)

        imgs = np.stack(list(reversed(img_list)), axis=0)
        masks = np.stack(list(reversed(mask_list)), axis=0)

        imgs_patch, masks_patch = random_crop(imgs, masks, self.patch_size, pos_prob=0.5)
        imgs_patch, masks_patch = self.tranform(imgs_patch, masks_patch)
        imgs_patch, masks_patch = imgs_patch[:, np.newaxis], masks_patch[:, np.newaxis]
        imgs_patch = torch.from_numpy(np.ascontiguousarray(imgs_patch))
        masks_patch = torch.from_numpy(np.ascontiguousarray(masks_patch))

        # with open(os.path.join(self.dataset_dir + '/labels/' + '{}.json'.format(self.seq_list[seq_idx])), 'r') as f:
        #     anno = json.load(f)
        # label = anno['label'][frame_idx]

        return imgs_patch, masks_patch

    def __len__(self):
        return len(self.seq_list) * self.frames_per_seq

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, test_dataset_name, frame_num=1):
        super(TestSetLoader).__init__()

        self.dataset_dir = dataset_dir + '/' + test_dataset_name + '/test'
        self.seq_list = sorted([int(f.split('.')[0]) for f in os.listdir(os.path.join(self.dataset_dir, 'labels'))])
        self.frame_num = frame_num
        self.frames_per_seq = 300
        
    def __getitem__(self, idx):
        seq_idx = idx // self.frames_per_seq
        frame_idx = idx % self.frames_per_seq

        img_list = []
        mask_list = []

        for i in range(self.frame_num):
            img = Image.open(self.dataset_dir + '/images/' + '{}/'.format(self.seq_list[seq_idx]) + '{}.png'.format(max(frame_idx - i, 0))).convert('I')
            mask = Image.open(self.dataset_dir + '/masks/' + '{}/'.format(self.seq_list[seq_idx]) + '{}.png'.format(max(frame_idx - i, 0)))

            img = Normalized(np.array(img, dtype=np.float32))
            mask = np.array(mask, dtype=np.float32) / 255.0

            img_list.append(img)
            mask_list.append(mask)

        imgs = np.stack(list(reversed(img_list)), axis=0)
        masks = np.stack(list(reversed(mask_list)), axis=0)
        
        t, h, w = imgs.shape
        imgs = PadImg(imgs, times=32)
        masks = PadImg(masks, times=32)
        
        imgs, masks = imgs[:,np.newaxis], masks[:,np.newaxis]
        
        imgs = torch.from_numpy(np.ascontiguousarray(imgs))
        masks = torch.from_numpy(np.ascontiguousarray(masks))
        return imgs, masks, [t,h,w], self.seq_list[seq_idx], frame_idx

    def __len__(self):
        return len(self.seq_list) * self.frames_per_seq


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input[:, :, ::-1]
            target = target[:, :, ::-1]
        if random.random()<0.5:
            input = input.transpose(0, 2, 1)
            target = target.transpose(0, 2, 1)
        return input, target


if __name__ == '__main__':

    db = TrainSetLoader(dataset_dir='your path', dataset_name='IDSMT', patch_size=256, frame_num=5)
    it = iter(db)
    for _ in range(5):
        imgs, masks = next(it)

