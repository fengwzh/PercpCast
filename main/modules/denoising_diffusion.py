import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal
from .layers.utils import warp, make_grid
from einops import rearrange, reduce, repeat
from .layers.evolution.phydnet import PhyDNet_Model
from omegaconf import OmegaConf
from accelerate import Accelerator
import random

op = 128

accelerator1 = Accelerator()


class RectifiedFlow():
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.N = num_steps

    @torch.no_grad()
    def get_train_tuple(self, z0=None, z1=None, trace=None, pre=None,cdf_exp=None):

        r = np.random.rand()  # 从[0, 1]中均匀采样一个随机数
        index = np.searchsorted(cdf_exp, r)
        index = index*100
        b = z0.shape[0]
        device = z0.device

        t_index =index//100
        z_index =random.randint(0, 100)
        target = z1[:, t_index:t_index + 1, :, :] - z0[:, t_index:t_index + 1, :, :]


        z_t = (1. -  z_index* 1 / 100) * z0[:, t_index:t_index + 1, :, :] + z_index  * 1 / 100 * z1[:, t_index:t_index + 1, :, :]
        # context = pre[:, t_index:t_index + 1, :, :]


        return z_t, z_index,t_index, target,z0[:, t_index:t_index + 1, :, :],z1[:, t_index:t_index + 1, :, :]

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None, context=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1. / 2
        traj = []  # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(2):

            t = (i)*50
            time = torch.full((z.shape[0],), t, device=z.device, dtype=torch.long)
            pred = self.model(z, time, context)
            z = z.detach().clone() + pred * dt

            traj.append(z.detach().clone())

        return traj

    def to_flattened_numpy(self,x):
        """Flatten a torch tensor `x` and convert it to numpy."""
        return x.detach().cpu().numpy().reshape((-1,))

    def from_flattened_numpy(self,x, shape):
        """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
        return torch.from_numpy(x.reshape(shape))

class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            channels=1,
            pred_mode="noise",
            aux_loss=True,
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.history_fn = None
        self.transform_fn = None
        assert pred_mode in ["noise", "pred_true"]
        self.pred_mode = pred_mode
        self.clip_noise = True
        self.otherlogs = {}
        self.aux_loss = aux_loss

        self.pred_length = 36


        self.loss1 = nn.L1Loss()

        self.rectified_flow = RectifiedFlow(model=denoise_fn, num_steps=3000)


        in_shape = (1, 128, 128)
        T_in = 13
        T_out = 36

        self.phydnet = PhyDNet_Model(in_shape, T_in, T_out, accelerator1.device)
        import lpips
        self.lpips_model = lpips.LPIPS(net='vgg')

        self.lpips_model = self.lpips_model.to(accelerator1.device)
        for p in self.lpips_model.parameters():
            p.requires_grad = False
    @torch.no_grad()
    def sample(self, init_frames, num_of_frames=12):

        B, T, C, H, W = init_frames.shape

        cur_seq = init_frames[:, :13, :, :, :].clone()

        motion_ = self.phydnet.inference(cur_seq)


        motion_ = motion_[:, :, 0, :, :]
        trace = motion_[:, :,  :, :].contiguous()
        # pre = motion_[:, :-1, :, :].contiguous()
        index = []
        for i in range(36):
            index.append(torch.full_like(motion_[:,0:1,:,:], i, device=trace.device, dtype=torch.long))
        pre = torch.cat(index, dim=1)
        # traj = self.rectified_flow.sample_ode(z0=motion_,motion_result=motion_, N=20)
        generated_frame = []
        # print(motion_.shape)

        combined_size = trace.size(0) * trace.size(1)

        x_viewed1 = trace.view(combined_size,1,  trace.size(2),  trace.size(3))

        combined_size2 = pre.size(0) * pre.size(1)

        x_viewed2 = pre.view(combined_size2,1,  pre.size(2),  pre.size(3))


        traj = self.rectified_flow.sample_ode(x_viewed1,  100,x_viewed2/36.0)[-1]
        # traj = self.rectified_flow.ode(x_viewed1, x_viewed2/36.0, 100)
        x_viewed2 = traj.view(trace.size(0),trace.size(1),  trace.size(2),  trace.size(3))
        # x_viewed2 = traj[-1].view(trace.size(0),trace.size(1),  trace.size(2),  trace.size(3))

        generated_frame.append(x_viewed2)

        frame = torch.cat(generated_frame, dim=1)
        # print(frame.shape)
        return frame
        # return motion_,frame,init_frames[:, 13:, 0, :, :]


    def p_losses1(self, x_start, motion_result, lossp,cdf_exp):

        # x_start1 = x_start[:, 13:, 0, :, :]
        # evo_result1 = motion_result

        z0 = motion_result[:, :,  :, :]
        z1 = x_start[:, 13:, 0, :, :]

        trace = motion_result[:, :,  :, :]
        loss1 = 0
        z_t, t,index, target,z_0,z_1 = self.rectified_flow.get_train_tuple(z0=z0, z1=z1, trace=trace,cdf_exp=cdf_exp)
        t = torch.full((z0.shape[0],), t, device=z0.device, dtype=torch.long)
        index = torch.full_like(z_t, index, device=z0.device, dtype=torch.long)
        pred = self.rectified_flow.model(z_t, t, index / 36.0)


        s_pred = z_0+pred*1
        losses = self.lpips_model(s_pred, z_1).mean()

        loss1 += 1*F.mse_loss(target, pred)

        loss = 1 * loss1 + 1 * lossp + 0.5*losses

        return t, loss.mean()

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7)).to(accelerator1.device)
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind, i, j] = 1
                ind += 1
        return constraints

    def forward(self, video,cdf_exp, prevideo=None):

        cur_seq = video[:, :, :, :, :].clone()
        self.constraints = self._get_constraints()

        lossp, motion_ = self.phydnet(cur_seq[:, :13, :, :, :], cur_seq[:, 13:, :, :, :], self.constraints, 0.0)
        # print(motion_.shape)
        motion_ = motion_[:, :, 0, :, :]

        evo_result = motion_
        loss = 0
        loss = self.p_losses1(video, motion_, lossp,cdf_exp)

        return loss
