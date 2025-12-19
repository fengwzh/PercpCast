import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import os
import os.path as osp

from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import cv2

# PIXEL_SCALE = 90.0
# THRESHOLDS = [12, 18, 24, 32]
# COLOR_MAP = ['lavender', 'indigo', 'mediumblue', 'dodgerblue', 'skyblue', 'cyan',
#              'olivedrab', 'lime', 'greenyellow', 'orange', 'red', 'magenta', 'pink', ]

COLOR_MAP = [[0, 0, 0],
              [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
              [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
              [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
              [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
              [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
              [0.9607843137254902, 0.9607843137254902, 0.0],
              [0.9294117647058824, 0.6745098039215687, 0.0],
              [0.9411764705882353, 0.43137254901960786, 0.0],
              [0.6274509803921569, 0.0, 0.0],
              [0.9058823529411765, 0.0, 1.0]]

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255
PIXEL_SCALE = 255.0
BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]
THRESHOLDS = (16, 74, 133, 160, 181, 219)

class Sevir(Dataset):
    def __init__(self, data_path, img_size, type='train', trans=None, seq_len=-1):
        super().__init__()

        self.pixel_scale = PIXEL_SCALE

        self.data_path = data_path
        self.img_size = img_size

        assert type in ['train', 'test', 'val']
        self.type = type if type != 'val' else 'test'
        with h5py.File(data_path, 'r') as f:
            # print(len(f[self.type]))
            if self.type=="train":
                self.all_len = len(f[self.type]) # 10000-3000 for train, 2000 for test, 1000 for val
            # self.all_data = f[self.type][()]

            elif self.type=="test":
                self.all_len = len(f[self.type][:2000])   # 10000-3000 for train, 2000 for test, 1000 for val
            #     self.all_data = f[self.type]
            # f.close()

        if trans is not None:
            self.transform = trans
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                # transforms.ToTensor(),
                # trans.Lambda(lambda x: x/255.0),
                # transforms.Normalize(mean=[0.5], std=[0.5]),
                # trans.RandomCrop(data_config["img_size"]),

            ])

    def __len__(self):
        return self.all_len

    def sample(self):
        index = np.random.randint(0, self.all_len)
        return self.__getitem__(index)

    def __getitem__(self, index):
        # with h5py.File(self.data_path, 'r') as f:
        #     # print(len(f[self.type]))
        #     if self.type=="train":
        #         self.all_len = int(len(f[self.type])*0.2)  # 10000-3000 for train, 2000 for test, 1000 for val
        #         self.all_data = f[self.type][-self.all_len:]
        #     elif self.type=="test":
        #         # self.all_len = 3220  # 10000-3000 for train, 2000 for test, 1000 for val
        #         self.all_data = f[self.type][:3220]
        #     f.close()
        with h5py.File(self.data_path, 'r') as f:
            # print(len(f[self.type]))
            if self.type=="train":
            # self.all_len = int(len(f[self.type]))  # 10000-3000 for train, 2000 for test, 1000 for val
                imgs = f[self.type][index][()]
            if self.type=="test":
            # self.all_len = int(len(f[self.type]))  # 10000-3000 for train, 2000 for test, 1000 for val
                imgs = f[self.type][index][()]
         # imgs = f[self.type][str(index)]  # numpy array: (25, 565, 784), dtype=uint8, range(0,70)

        frames = torch.from_numpy(imgs).float().squeeze()
            # frames = frames / self.pixel_scale
            # frames = self.transform(frames)
        # print(frames.mean())
        return frames.unsqueeze(1)  # (25,1,128,128)


# def gray2color(img):
#     cmap = colors.ListedColormap(COLOR_MAP)
#     norm = colors.BoundaryNorm(BOUNDS, cmap.N)
#     return cmap(norm(img))

def gray2color(image):
    # 定义颜色映射和边界
    cmap = colors.ListedColormap(COLOR_MAP)
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))

    return colored_image


def vis_res(pred_seq, gt_seq, save_path, save_grays=False, do_hmf=False, save_colored=False):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
    pred_seq = pred_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(osp.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):
            # plt.imsave(osp.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmin=0, vmax=1)
            # plt.imsave(osp.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmin=0, vmax=1)
            cv2.imwrite(osp.join(save_path, 'pred', f'{i}.png'), (pred * PIXEL_SCALE).astype(np.uint8))
            cv2.imwrite(osp.join(save_path, 'targets', f'{i}.png'), (gt * PIXEL_SCALE).astype(np.uint8))

    pred_seq = pred_seq * PIXEL_SCALE
    pred_seq = pred_seq.astype(np.int16)
    gt_seq = gt_seq * PIXEL_SCALE
    gt_seq = gt_seq.astype(np.int16)

    colored_pred = np.array([gray2color(pred_seq[i]) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt = np.array([gray2color(gt_seq[i]) for i in range(len(gt_seq))], dtype=np.float64)

    if save_colored:
        os.makedirs(osp.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets_colored'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(osp.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(osp.join(save_path, 'targets_colored', f'{i}.png'), gt)

    grid_pred = np.concatenate([
        np.concatenate([i for i in colored_pred], axis=-2),
    ], axis=-3)
    grid_gt = np.concatenate([
        np.concatenate([i for i in colored_gt], axis=-2, ),
    ], axis=-3)

    grid_concat = np.concatenate([grid_pred, grid_gt], axis=-3, )
    plt.imsave(osp.join(save_path, 'all.png'), grid_concat)

    clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
    clip.write_gif(osp.join(save_path, 'pred.gif'), fps=4, verbose=False)
    clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
    clip.write_gif(osp.join(save_path, 'targets.gif'), fps=4, verbose=False)

    if do_hmf:
        def hit_miss_fa(y_true, y_pred, thres):
            mask = np.zeros_like(y_true)
            mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
            mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
            mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
            mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
            return mask

        grid_pred = np.concatenate([
            np.concatenate([i for i in pred_seq], axis=-1),
        ], axis=-2)
        grid_gt = np.concatenate([
            np.concatenate([i for i in gt_seq], axis=-1),
        ], axis=-2)

        hmf_mask = hit_miss_fa(grid_pred, grid_gt, thres=THRESHOLDS[1])
        plt.axis('off')
        plt.imsave(osp.join(save_path, 'hmf.png'), hmf_mask, cmap=colors.ListedColormap(HMF_COLORS))


if __name__ == '__main__':
    dataset = Sevir('/mnt/sda/merged1.h5', 128,type='test')
    sample1 = dataset.sample()
    # sample2 = dataset.sample()

    # vis_res(sample1.numpy(), sample2.numpy(), save_path='./test', save_grays=True, do_hmf=True)

    print(len(dataset))