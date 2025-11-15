import torch
from torch import fft
import torchkbnufft as tkbn
import numpy as np
import matplotlib.pyplot as plt
import os
# import mapvbvd
import sigpy.mri as mr
from typing import List, Tuple, Union, Optional
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

device = torch.device("cuda")
GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2)) 

def gen_traj(theta, spoke_length, spoke_num, ind=0):
    angles = theta * torch.arange(ind, ind+spoke_num, dtype=torch.float32, device=device).unsqueeze_(1)

    pos = torch.linspace(-np.pi, np.pi, spoke_length, device=device).unsqueeze_(0)
    kx = torch.mm(torch.cos(angles), pos)
    ky = torch.mm(torch.sin(angles), pos)

    return torch.stack((kx.flatten(), ky.flatten()))


@torch.jit.script
def fftnc(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    N-dim centered FFT

    :param x: input N-dim Tensor (CPU/GPU)
    :param dim: run FFT in given dim
    :return: output N-dim Tensor (CPU/GPU)
    """
    device = x.device
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.ifftshift(fft.fftn(fft.fftshift(x, dim=dim), dim=dim, norm='ortho'), dim=dim)


@torch.jit.script
def ifftnc(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    N-dim centered iFFT

    :param x: input N-dim Tensor (CPU/GPU)
    :param dim: run iFFT in given dim
    :return: output N-dim Tensor (CPU/GPU)
    """
    device = x.device
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=dim), dim=dim), dim=dim)




def cal_metrics(recon_img, gt_img):
    
    if isinstance(recon_img, torch.Tensor):
        recon_img = recon_img.detach().cpu().numpy()
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    psnr_lst = []
    ssim_lst = []
    data_range = gt_img.max() - gt_img.min()

    for i in range(recon_img.shape[0]):
        psnr_lst.append(psnr(recon_img[i, :, :], gt_img[i, :, :], data_range=data_range))
        ssim_lst.append(ssim(recon_img[i, :, :], gt_img[i, :, :], data_range=data_range))

    return np.mean(psnr_lst), np.mean(ssim_lst)


def save_as_gif(array, filename, min_value=0, max_value=1):
    from PIL import Image
    images = []
    array = np.clip(array, min_value, max_value)  # Ensure values are in [min_value, max_value]
    array = (array - min_value) / (max_value - min_value)  # Normalize to [0, 1]
    for i in range(array.shape[0]):
        img_slice = (array[i] * 255).astype(np.uint8)  # Scale to 0-255
        images.append(Image.fromarray(img_slice))
    images[0].save(filename, save_all=True, append_images=images[1:], duration=100, loop=0)


def plot_dvf_frames(dvf, frames, interval, save_path):
    n_frames = len(frames)
    n_cols = 4
    n_rows = (n_frames + n_cols - 1) // n_cols  # integer division to get number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten() if n_frames > 1 else [axes]

    for idx, n_frame in enumerate(frames):
        dvf_frame = dvf[n_frame, :, :, :]
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.5, 0.5, dvf_frame.shape[0]),
            np.linspace(-0.5, 0.5, dvf_frame.shape[1]),
            indexing="ij",
        )
        grid_coords = np.stack((grid_x, grid_y), axis=-1)
        # Debug prints (optional)
        # print(grid_coords.shape)
        warped_grid = grid_coords + dvf_frame
        # print(grid_coords[:, 0])
        ax = axes[idx]
        ax.plot(warped_grid[:, 1::interval, 1], warped_grid[:, 1::interval, 0], linewidth=0.5, color='blue')
        ax.plot(warped_grid[1::interval, :, 1].T, warped_grid[1::interval, :, 0].T, linewidth=0.5, color='blue')
        
        # Reverse the directions of the y axis.
        ax.invert_yaxis()
        ax.set_title(f'DVF Frame {n_frame + 1}')

    # Turn off any unused subplots.
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close(fig)



