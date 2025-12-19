from data import get_dataset
import config
import argparse
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
from accelerate import Accelerator
import torch
parser = argparse.ArgumentParser(description="values from bash script")
args = parser.parse_args()
from accelerate import DistributedDataParallelKwargs
from torch.utils.data.dataloader import default_collate
from pytorch_lightning import seed_everything

seed = 3024
seed_everything(seed, workers=True)
def _load_data():
    train_batch_size = 50
    test_batch_size = 20
    train_data, valid_data, test_data, vis_norm_fn,  THRESHOLDS, PIXEL_SCALE = get_dataset.get_dataset(
        data_name= 'sevir',
        data_path=path,
        img_size=128,
        seq_len=25,
        batch_size=train_batch_size,
    )
    print(len(train_data),len(test_data))
    def transposed_collate(batch):
        batch = filter(lambda img: img is not None, batch)
        collated_batch = default_collate(list(batch))
        transposed_batch = collated_batch.transpose_(0, 1)
        return transposed_batch
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        num_workers=40,
        shuffle=True,
        # or transposed_collate, which aims to transpose B and T dim
        collate_fn=transposed_collate,
        pin_memory=False,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=40,
        collate_fn=transposed_collate,
        drop_last=False,
    )
    return train_loader, test_loader

def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


def main():
    print("start")
    train_data, val_data= _load_data()

    denoise_model = Unet(
        dim= 64,
        channels=1,
    )

    diffusion = GaussianDiffusion(

        denoise_fn=denoise_model,
        pred_mode=config.pred_mode,
        clip_noise=config.clip_noise,
        timesteps=config.iteration_step,
        loss_type=config.loss_type,
        aux_loss=False,
    )


    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    diffusion, config.optimizer, train_data,val_data = accelerator.prepare(
     diffusion, config.optimizer, train_data,val_data)

    trainer = Trainer(
        diffusion_model=diffusion,
        train_dl=train_data,
        val_dl=val_data,
        sample_num_of_frame=config.init_num_of_frame + 1,
        init_num_of_frame=config.init_num_of_frame,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        ema_decay=config.ema_decay,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        step_start_ema=config.ema_start_step,
        update_ema_every=config.ema_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{config.model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{config.model_name}/"),
        model_name=config.model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        accelerator=accelerator,
    )

    if config.load_model:
        trainer.load(load_step=config.load_step)

    trainer.train()


if __name__ == "__main__":

    main()

