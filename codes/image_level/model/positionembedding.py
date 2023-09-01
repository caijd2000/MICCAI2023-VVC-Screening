import torch
import numpy as np
import math
def Position_Embedding( normalize=True, scale=None, sq=None):
    num_pos_feats = 1024 #channel/2
    temperature = 10000
    if scale is None:
        scale = 2 * math.pi

    x = list(range(sq.shape[3]))       #sq is N, C, H, W
    y = list(range(sq.shape[2]))

    x_embed = torch.from_numpy(np.matrix([x for _ in range(sq.shape[2])])).unsqueeze(0)+1
    y_embed = torch.from_numpy(np.matrix([y for _ in range(sq.shape[3])])).T.unsqueeze(0)+1

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos