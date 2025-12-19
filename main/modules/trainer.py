import copy
import torch
import os
import shutil
import torch.distributed as dist
from pathlib import Path
from torch.optim import Adam, AdamW,SGD
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from sevir.sevir_vis_seq import save_example_vis_results
from sevir.sevir_cmap import get_cmap, VIL_COLORS, VIL_LEVELS
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from ema_pytorch import EMA
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F


# from diffusers.optimization import get_scheduler
from timm.scheduler import CosineLRScheduler
import time
import numpy as np
import logging
import lpips
import torch
import cv2
import os
import matplotlib.pyplot as plt
import tqdm
from einops import rearrange
import warnings
warnings.filterwarnings("ignore")
plt.switch_backend('agg')
np.seterr(divide='ignore', invalid='ignore')


def print_log(message, is_main_process=True):
    if is_main_process:
        print(message)
        logging.info(message)

class PaperEvaluator(object):
    def __init__(self, seq_len=20, value_scale=255, thresholds=[16, 74, 133, 160, 181, 219], **kwargs):
        self.metrics = {}
        self.thresholds = thresholds
        for threshold in self.thresholds:
            self.metrics[threshold] = {
                "hits": [],
                "misses": [],
                "falsealarms": [],
                "correctnegs": [],

                "hits44": [],
                "misses44": [],
                "falsealarms44": [],
                "correctnegs44": [],

                "hits16": [],
                "misses16": [],
                "falsealarms16": [],
                "correctnegs16": [],
            }
        self.losses = {
            "mse": [],
            "mae": [],
            "rmse": [],
            "psnr": [],
            "ssim": [],
            "crps": [],
            "lpips": [],
        }
        self.seq_len = seq_len
        self.total = 0
        self.value_scale = value_scale

    def float2int(self, arr):

        x = arr.clip(0.0, 1.0)
        x = x * self.value_scale
        x = x.astype(np.uint16)
        return x

    def evaluate(self, true_batch, pred_batch):
        # [batch_size, seq_len, 64, 64], data_range [0.,1.]
        if isinstance(pred_batch, torch.Tensor):
            pred_batch = pred_batch.detach().cpu().numpy()
            true_batch = true_batch.detach().cpu().numpy()

        if not (true_batch.max() <= 1.0 and true_batch.min() >= 0.0):
            print_log(f"WARNING:: data max: {true_batch.max()}, min: {true_batch.min()}")

        pred_batch = pred_batch.clip(0.0, 1.0)
        true_batch = true_batch.clip(0.0, 1.0)

        assert pred_batch.shape == true_batch.shape, f"pred_batch.shape: {pred_batch.shape}, true_batch.shape: {true_batch.shape}"

        batch_size, seq_len = true_batch.shape[:2]

        pred = self.float2int(pred_batch)
        gt = self.float2int(true_batch)

        for threshold in self.thresholds:
            for b in range(batch_size):
                seq_hit, seq_miss, seq_falsealarm, seq_correctneg = [], [], [], []
                for t in range(seq_len):
                    hit, miss, falsealarm, correctneg = self.cal_frame(gt[b][t], pred[b][t], threshold)
                    seq_hit.append(hit)
                    seq_miss.append(miss)
                    seq_falsealarm.append(falsealarm)
                    seq_correctneg.append(correctneg)

                self.metrics[threshold]["hits"].append(seq_hit)
                self.metrics[threshold]["misses"].append(seq_miss)
                self.metrics[threshold]["falsealarms"].append(seq_falsealarm)
                self.metrics[threshold]["correctnegs"].append(seq_correctneg)

        self.total += batch_size

    def cal_frame(self, obs, sim, threshold):
        obs = np.where(obs >= threshold, 1, 0)
        sim = np.where(sim >= threshold, 1, 0)

        # True positive (TP)
        hits = np.sum((obs == 1) & (sim == 1))

        # False negative (FN)
        misses = np.sum((obs == 1) & (sim == 0))

        # False positive (FP)
        falsealarms = np.sum((obs == 0) & (sim == 1))

        # True negative (TN)
        correctnegatives = np.sum((obs == 0) & (sim == 0))

        return hits, misses, falsealarms, correctnegatives


    def done(self):
        res_dict = {

        }

        avg_csi, avg_far, avg_pod, avg_hss = [], [], [], []
        avg_csi44, avg_csi16 = [], []
        for threshold in self.thresholds:
            hits = np.array(self.metrics[threshold]["hits"])
            misses = np.array(self.metrics[threshold]["misses"])
            falsealarms = np.array(self.metrics[threshold]["falsealarms"])
            correctnegs = np.array(self.metrics[threshold]["correctnegs"])

            # remove nan
            hits = np.nan_to_num(hits)
            misses = np.nan_to_num(misses)
            falsealarms = np.nan_to_num(falsealarms)
            correctnegs = np.nan_to_num(correctnegs)

            # first cal method

            csi1 = np.mean(hits, axis=0) / (
                        np.mean(hits, axis=0) + np.mean(misses, axis=0) + np.mean(falsealarms, axis=0))
            far1 = np.mean(falsealarms, axis=0) / (np.mean(hits, axis=0) + np.mean(falsealarms, axis=0))
            pod1 = np.mean(hits, axis=0) / (np.mean(hits, axis=0) + np.mean(misses, axis=0))
            hss1 = 2 * (np.mean(hits, axis=0) * np.mean(correctnegs, axis=0) - np.mean(misses, axis=0) * np.mean(
                falsealarms, axis=0)) / ((np.mean(hits, axis=0) + np.mean(misses, axis=0)) * (
                        np.mean(misses, axis=0) + np.mean(correctnegs, axis=0)) + (
                                                     np.mean(hits, axis=0) + np.mean(falsealarms, axis=0)) * (
                                                     np.mean(falsealarms, axis=0) + np.mean(correctnegs, axis=0)))

            csi1 = np.nan_to_num(csi1)
            far1 = np.nan_to_num(far1)
            pod1 = np.nan_to_num(pod1)
            hss1 = np.nan_to_num(hss1)

            avg_csi.append(np.mean(csi1))
            avg_far.append(np.mean(far1))
            avg_pod.append(np.mean(pod1))
            avg_hss.append(np.mean(hss1))

        print_log(
            f"[ avg_csi ] : {np.mean(avg_csi)}; [ avg_far ] : {np.mean(avg_far)}; [ avg_pod ] : {np.mean(avg_pod)}; [ avg_hss] : {np.mean(avg_hss)}")
        res_dict['csi'] = np.nan_to_num(np.mean(avg_csi))


        return np.mean(avg_csi)

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_dl,
            val_dl,
            sample_num_of_frame,
            init_num_of_frame,
            scheduler_function,
            accelerator,
            ema_decay=0.995,
            train_lr=1e-4,
            train_num_steps=1000000,
            scheduler_checkpoint_step=10000,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder="./results",
            gradient_accumulate_every=1,
            tensorboard_dir="./tensorboard_logs/diffusion-video/",
            model_name="model",
            val_num_of_batch=1,
            max_grad_norm=1.0,
            optimizer="adam",

    ):
        super().__init__()
        self.model = diffusion_model

        self.update_ema_every = update_ema_every
        self.sample_num_of_frame = sample_num_of_frame
        self.val_num_of_batch = val_num_of_batch

        self.step_start_ema = 2000
        self.save_and_sample_every = save_and_sample_every
        self.max_grad_norm = max_grad_norm
        self.train_num_steps = train_num_steps

        self.train_dl_class = train_dl
        self.val_dl_class = val_dl
        self.l1 = len(train_dl)

        self.opt = AdamW(self.model.parameters(), lr=3e-4)
        self.epcoh = int(400/4)
        self.warmup = int(self.epcoh * 0.2/4)

        self.scheduler = CosineLRScheduler(self.opt,
                                            t_initial=self.epcoh,
                                            cycle_decay=0.5,
                                            lr_min=1e-6,
                                            t_in_epochs=True,
                                            warmup_t=self.warmup,
                                            warmup_lr_init=1e-4,
                                            cycle_limit=1,
                                            )

        self.gradient_accumulate_every = gradient_accumulate_every
        self.step = 0
        self.device = accelerator.device
        self.init_num_of_frame = 13
        self.scheduler_checkpoint_step = 1500

        self.results_folder = "results",

        self.model_name = model_name

        if os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)

        self.accelerator = accelerator



    def save(self,epoch):
        if str(self.device) == "cuda:0":
            data = {
                "step": epoch,
                "model": self.accelerator.get_state_dict(self.model),
            }
            idx = epoch
            torch.save(data,  str(idx) + ".pt")

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str("model_1" + str(idx) + ".pt"),
            map_location=lambda storage, loc: storage,
        )
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])


    def train(self):
        self.opt, self.scheduler = self.accelerator.prepare(self.opt, self.scheduler)
        loss_fn = lpips.LPIPS(net="vgg").cuda()  # 可以选择 'alex', 'vgg' 或 'squeeze'

        M = np.arange(36)  # 样本索引 0 到 35
        # k = 0.05  # 权重增长因子
        k = 0.05  # 权重增长因子
        T_start = 50  # 初始温度，足够高以接近平滑分布
        T_end = 0.5  # 最终温度，控制后期集中程度
        t_max = 100  # 最大时间步数

        for epoch in range(0, self.epcoh):
            if self.accelerator.is_main_process:
                T1 = time.time()
            self.model.train()
            t = epoch
            T_exp = T_start * (T_end / T_start) ** (t / t_max)
            weights = np.exp(k * M)
            weights_exp = weights ** (1 / T_exp)
            p_exp = weights_exp / np.sum(weights_exp)
            cdf_exp = np.cumsum(p_exp)

            for step, batch in enumerate(self.train_dl_class):
                # print(batch.shape)
                batch = torch.permute(batch, (1, 0, 2, 3, 4))
                data = batch.to(self.device)
                t,loss,pred = self.model(data,cdf_exp)

                self.accelerator.backward(loss)

                self.accelerator.wait_for_everyone()
                if step % 1 ==0:
                    self.opt.step()
                    # if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step(epoch + step / self.l1/1)
                    self.opt.zero_grad()
                self.accelerator.wait_for_everyone()
                # if step>3:
                #     break

            if self.accelerator.is_main_process:
                T2 = time.time()
                print('times:',((T2 - T1) * 1000))
                print("train",epoch,step,loss.item())
                # self.save(epoch)
                loss = 0
                self.model.eval()
                l = []
                l1 = []
                total = 0
                score_ssim = []
                score_fid = []
                score_psnr = []
                score_mae = []
                score_lpips = []
                score_mse = []
                evaluator = PaperEvaluator()
                evaluator1 = [PaperEvaluator(thresholds=[o]) for o in [16, 74, 133, 160, 181, 219]]
                for iii, batch in enumerate(self.val_dl_class):

                    batch = torch.permute(batch, (1, 0, 2, 3, 4))

                    videos = self.model.module.sample(
                        batch,
                        self.sample_num_of_frame,
                    )

                    gt = batch[:, :, 0]
                    pd = videos.clamp(0, 1)
                    evaluator.evaluate(gt[:, 13:, :, :], pd)
                    for t1 in range(36):
                        score, diff = ssim(gt[:, 13 + t1:13 + t1 + 1, :, :][0, 0].cpu().numpy(),
                                           pd[:, t1:t1 + 1, :, :][0, 0].cpu().numpy(), full=True)
                        score_ssim.append(score.astype(np.float16))
                        data_range = gt[:, 13 + t1:13 + t1 + 1, :, :].max() - gt[:, 13 + t1:13 + t1 + 1, :, :].min()
                        mse = np.mean((gt[:, 13 + t1:13 + t1 + 1, :, :][0, 0].cpu().numpy() - pd[:, t1:t1 + 1, :, :][
                            0, 0].cpu().numpy()) ** 2)
                        score_mse.append(mse.astype(np.float16))
                        epsilon = 1e-10  # 一个很小的数，防止除零
                        mse += epsilon
                        psnr_value = 10 * np.log10((1 ** 2) / mse)
                        score_psnr.append(psnr_value)
                        score_mae.append(F.l1_loss(gt[:, 13 + t1:13 + t1 + 1, :, :][:, 0], pd[:, t1:t1 + 1, :, :][:, 0],
                                                   reduction='mean').cpu().numpy().astype(np.float16))
                        with torch.no_grad():
                            lpip = loss_fn(gt[:, 13 + t1:13 + t1 + 1, :, :], pd[:, t1:t1 + 1, :, :])
                            score_lpips.append(lpip.cpu().numpy().astype(np.float16))
                    [evaluator1[o].evaluate(gt[:, 13:, :, :], pd) for o in range(6)]
                csi = [evaluator1[o].done() for o in range(6)]
                r = evaluator.done()

                print("CSI", r, "SSIM", np.mean(score_ssim), "PSNR", np.mean(score_psnr), "MAE", np.mean(score_mae),
                      "MSE", np.mean(score_mse), "LPIPS", np.mean(score_lpips))

                with open("res.txt", "a") as f:
                    f.write("step" + str(epoch) + " r " + str(r) + "CSI" + str(r) + "SSIM" + str(
                        np.mean(score_ssim)) + "PSNR" + str(np.mean(score_psnr)) + "MAE" + str(np.mean(score_mae)) +
                            "MSE" + str(np.mean(score_mse)) + "LPIPS" + str(np.mean(score_lpips)) + "\n")
                    # for item in l1:
                if r > 0.265:
                    self.save(epoch)
                    print("save model")
            self.accelerator.wait_for_everyone()

        print("training completed")
