import torch

def post_precoss_fn(seqs, value_scale, norm=False):
    if isinstance(seqs, torch.Tensor):
        seqs = seqs.detach().cpu().numpy()

    if norm:
        seqs = seqs.clip(-1.0, 1.0)
        seqs = (seqs + 1.0) / 2.0
    # [B, T, C, H, W]
    seqs = seqs.clip(0., 1.0)
    seqs = seqs * value_scale
    # seqs = seqs.astype(np.int16)
    return seqs


# @staticmethod
def get_dataset(data_name, data_path, img_size, seq_len, **kwargs):
    dataset_name = data_name.lower()

    train = val = test = None
    if data_name == 'sevir':
        from .dataset_sevir_new import Sevir, vis_res, THRESHOLDS, PIXEL_SCALE
        train = Sevir(data_path, type='train', img_size=img_size)
        test = Sevir(data_path, type='test', img_size=img_size)


    return train, val, test, vis_res, THRESHOLDS, PIXEL_SCALE, 