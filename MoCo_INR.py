import torch
import os
import sys
import datetime
import json
import argparse

import numpy as np
import tinycudann as tcnn
import torchkbnufft as tkbn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.optim import lr_scheduler
from scipy import io
import imageio as imgio
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import SimpleITK as sitk
import torch.nn as nn

from utils import *


class RelL2Loss(torch.nn.Module):
    def __init__(self, eps=1e-4):
        super(RelL2Loss, self).__init__()
        self.eps = eps

    def forward(self, input, label):
        loss = (label.real - input.real) ** 2 / (input.real.detach() ** 2 + self.eps) + \
               (label.imag - input.imag) ** 2 / (input.imag.detach() ** 2 + self.eps)
        return loss.mean()

class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        loss_fn = torch.nn.MSELoss()
        self.loss_fn = loss_fn

    def forward(self, input, label):
        input = torch.view_as_real(input)
        label = torch.view_as_real(label)
        return self.loss_fn(input, label)

class L1Loss(torch.nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        loss_fn = torch.nn.L1Loss()
        self.loss_fn = loss_fn

    def forward(self, input, label):
        input = torch.view_as_real(input)
        label = torch.view_as_real(label)
        return self.loss_fn(input, label)

class SpatialTV(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(SpatialTV, self).__init__()
        self.weight = weight

    def forward(self, x):
        # Compute the total variation loss
        dx = torch.abs(x[:, 1:] - x[:, :-1]).mean()
        dy = torch.abs(x[1:, :] - x[:-1, :]).mean()

        return self.weight * (dx + dy)

class LaplacianLoss2d(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(LaplacianLoss2d, self).__init__()
        self.weight = weight

        self.kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # x shape (grid_size, grid_size, n_frame, 2)
        x = x.permute(2, 3, 0, 1)  # (n_frame, 2, grid_size, grid_size)
        N, C, H, W = x.shape
        kernel = self.kernel.to(x.device).repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
        laplacian = F.conv2d(x, kernel, padding=1, groups=2)
        loss = torch.mean(torch.abs(laplacian))  # Mean absolute error

        return self.weight * loss

class MoCoINR:
    def __init__(self, config, gt_img=None, smap=None, kdata=None, mask_tensor=None):
        """
        Args:
            config: 包含所有配置参数的字典或配置对象
        """

        self.train_config = config['train']
        self.evaluation_config = config['evaluation']
        self.net_config = config['net']

        self.device = torch.device(f"cuda:{self.train_config['gpu']}" if torch.cuda.is_available() else "cpu")

        # 初始化网络和优化器
        self._init_networks()
        self._init_optimizers()
        self._init_loss_functions()
        
        # 数据相关
        self.gt_img = gt_img
        self.smap = smap
        self.kdata = kdata
        self.mask_tensor = mask_tensor
        self.img_mask = torch.ones_like(self.gt_img, dtype=torch.float32)
        self.img_mask[self.gt_img == 0] = 0

        self.n_frame = self.gt_img.shape[0]
        self.n_coil = self.kdata.shape[1]
        self.H, self.W = self.gt_img.shape[-2], self.gt_img.shape[-1]

        self.lamda = self.train_config['lamda']
        self.save_path = self.evaluation_config['save_path']

        for sub_dir in ['cano', 'dynamic']:
            if not os.path.exists(f'{self.save_path}/{sub_dir}'):
                os.makedirs(f'{self.save_path}/{sub_dir}')

        self.xy_coord, self.xyt_coord = self._init_train_coord()


        
    def _init_networks(self):

        self.DVF_net_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config= self.net_config["dvf_net_enc"],).to(self.device)
        
        self.DVF_net_dec = nn.Sequential(
        nn.Conv2d(self.DVF_net_enc.n_output_dims, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 2, kernel_size=3, padding=1)).to(self.device)
        
        self.CANO_net_enc = tcnn.Encoding(
            n_input_dims=2,
            encoding_config= self.net_config["cano_net_enc"],).to(self.device)
        
        self.CANO_net_dec = nn.Sequential(
            nn.Conv2d(self.CANO_net_enc.n_output_dims, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        ).to(self.device)

    def _init_train_coord(self):

        n_frame = self.n_frame

        """初始化xyt坐标"""
        xs = torch.fft.fftshift(torch.fft.fftfreq(self.H)).to(self.device)
        ys = torch.fft.fftshift(torch.fft.fftfreq(self.W)).to(self.device)
        ts = torch.fft.fftshift(torch.fft.fftfreq(self.W))[(self.W - n_frame)//2 : (self.W + n_frame)//2].to(self.device)

        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy_coord = torch.stack((xv, yv), axis = -1)

        xv, yv, tv = torch.meshgrid([xs, ys, ts], indexing="ij")
        xyt_coord = torch.stack((xv, yv, tv), axis = -1)    

        return xy_coord, xyt_coord


    def _init_optimizers(self):
        """初始化优化器"""

        self.optimizer = torch.optim.Adam([
                                {'params': self.DVF_net_enc.parameters(), 'lr': self.train_config["lr"]},
                                {'params': self.DVF_net_dec.parameters(), 'lr': self.train_config["lr"]},
                                {'params': self.CANO_net_enc.parameters(), 'lr': self.train_config["lr"]},
                                {'params': self.CANO_net_dec.parameters(), 'lr': self.train_config["lr"]}])
        
    def _init_loss_functions(self):
        """initialize loss functions"""
        if self.train_config["loss_fn"] == 'relL2':
            self.dc_loss_fn = RelL2Loss()
        elif self.train_config["loss_fn"] == 'l2':
            self.dc_loss_fn = L2Loss()
        elif self.train_config["loss_fn"] == 'l1':
            self.dc_loss_fn = L1Loss()
        
        self.warped_reg_fn = L2Loss()
        self.tv_loss_fn = SpatialTV()
        self.lap_loss_fn = LaplacianLoss2d()


    def train(self):
        """Whole Training Pipeline"""
        epoch_loop = tqdm(range(self.train_config["epoch"]), 
                          total=self.train_config["epoch"], 
                          ncols=120)
        
        for epoch in epoch_loop:
            loss_dict = self.train_step(epoch)

            epoch_loop.set_postfix(loss = f"{loss_dict['loss']:.6f}")
            
            if self.evaluation_config["flag"]:
                if (epoch + 1) % self.evaluation_config["summary_epoch"] == 0:
                    recon = self.evaluate(epoch)
    

        recon = self.evaluate(epoch)
        self.save_model(f'{self.save_path}')  
        save_as_gif(torch.abs(recon).numpy(), 
                    f'{self.save_path}/recon.gif', 0, 0.6)

        return recon 

    def train_step(self, epoch):
        """Single Training Step"""
        self.optimizer.zero_grad()
        
        # Select how many frames to train in this iteration
        if self.train_config["train_frame"] == 0:
            xyt_batch = self.xyt_coord
            mask_batch = self.mask_tensor
            target_kdata = self.kdata
        else:
            start_frame = np.random.randint(0, self.n_frame - self.train_config["train_frame"] + 1)
            end_frame = start_frame + self.train_config["train_frame"]
            xyt_batch = self.xyt_coord[:, :, start_frame:end_frame]
            mask_batch = self.mask_tensor[start_frame:end_frame]
            target_kdata = self.kdata[start_frame:end_frame]

        dvf_enc_mask = torch.ones(size = (1, self.DVF_net_enc.n_output_dims)).to(self.device)
        cano_enc_mask = torch.ones(size = (1, self.CANO_net_enc.n_output_dims)).to(self.device)

        if self.train_config['C2F']:
            C2F_config = self.train_config['C2F_config']
            if epoch <= C2F_config['stage_1']:
                dvf_enc_mask[:, int(self.net_config["dvf_net_enc"]["n_features_per_level"] * 6):] = 0
                cano_enc_mask[:, int(self.net_config["cano_net_enc"]["n_features_per_level"] * 8):] = 0
            elif epoch <= C2F_config['stage_2']:
                dvf_enc_mask[:, int(self.net_config["dvf_net_enc"]["n_features_per_level"] * 8):] = 0
                cano_enc_mask[:, int(self.net_config["cano_net_enc"]["n_features_per_level"] * 10):] = 0
            else: 
                pass

        dvf_feat = self.DVF_net_enc(xyt_batch.reshape(-1,3).to(device)).float() * dvf_enc_mask
        dvf_feat = dvf_feat.reshape(self.H, 
                                    self.W, 
                                    -1, dvf_feat.shape[-1]).float()
        dvf_feat = dvf_feat.permute(2, 3, 0, 1)

        xy_delta = self.DVF_net_dec(dvf_feat).float()  # (n_frame, grid_size, grid_size, 2)
        xy_delta = xy_delta.permute(2, 3, 0, 1)  # (grid_size, grid_size, n_frame, 2)
        # xy_delta = xy_delta.reshape(grid_size, grid_size, -1, 2).float()

        deform_grid = xyt_batch[..., :2] + xy_delta

        cano_feat = self.CANO_net_enc(deform_grid.reshape(-1, 2).to(device)).float() * cano_enc_mask
        cano_feat = cano_feat.reshape(self.H, self.W, -1, cano_feat.shape[-1]).float()
        cano_feat = cano_feat.permute(2, 3, 0, 1)

        est_dynamic = self.CANO_net_dec(cano_feat).float()  # (H, W, 2)
        est_dynamic = est_dynamic.permute(0, 2, 3, 1).contiguous()  # (frame_train, H, W, 2)
        est_dynamic = torch.view_as_complex(est_dynamic)  # (frame_train, H, W)
        # print(est_dynamic.shape)

        est_dynamic_multi = est_dynamic.unsqueeze(1) * self.smap.unsqueeze(0)  
        
        est_kdata = fftnc(est_dynamic_multi, dim = (-2, -1)) 
        mask_batch = mask_batch.unsqueeze(1).repeat(1, self.n_coil, 1, 1)  # (n_frame, n_coil, n_kx, n_ky)

        dc_loss = self.dc_loss_fn(est_kdata[mask_batch==1], target_kdata[mask_batch==1])

        tv_loss = self.tv_loss_fn(xy_delta)
        lap_loss = self.lap_loss_fn(xy_delta)
        sparse_loss = torch.mean(torch.abs(xy_delta))  # Sparse loss


        loss = dc_loss + self.lamda * (tv_loss + lap_loss + sparse_loss)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def evaluate(self, epoch):
        """评估和保存结果"""
        with torch.no_grad():
            self.DVF_net_enc.eval()
            self.DVF_net_dec.eval()
            self.CANO_net_enc.eval()
            self.CANO_net_dec.eval()

            dvf_enc_mask = torch.ones(size = (1, self.DVF_net_enc.n_output_dims)).to(device)
            cano_enc_mask = torch.ones(size = (1, self.CANO_net_enc.n_output_dims)).to(device)

            if self.train_config['C2F']:
                C2F_config = self.train_config['C2F_config']
                if epoch <= C2F_config['stage_1']:
                    dvf_enc_mask[:, int(self.net_config["dvf_net_enc"]["n_features_per_level"] * 6):] = 0
                    cano_enc_mask[:, int(self.net_config["cano_net_enc"]["n_features_per_level"] * 8):] = 0
                elif epoch <= C2F_config['stage_2']:
                    dvf_enc_mask[:, int(self.net_config["dvf_net_enc"]["n_features_per_level"] * 8):] = 0
                    cano_enc_mask[:, int(self.net_config["cano_net_enc"]["n_features_per_level"] * 10):] = 0
                else: 
                    pass

            dvf_feat = self.DVF_net_enc(self.xyt_coord.reshape(-1,3).to(device)).float() * dvf_enc_mask
            dvf_feat = dvf_feat.reshape(self.H, self.W, -1, dvf_feat.shape[-1]).float()
            dvf_feat = dvf_feat.permute(2, 3, 0, 1)

            xy_delta = self.DVF_net_dec(dvf_feat).float()
            xy_delta = xy_delta.permute(2, 3, 0, 1)
            deform_grid = self.xyt_coord[..., :2] + xy_delta

            cano_feat = self.CANO_net_enc(deform_grid.reshape(-1, 2).to(device)).float() * cano_enc_mask
            cano_feat = cano_feat.reshape(self.H, self.W, -1, cano_feat.shape[-1]).float()
            cano_feat = cano_feat.permute(2, 3, 0, 1)
            est_dynamic = self.CANO_net_dec(cano_feat).float()  # (H, W, 2)
            est_dynamic = est_dynamic.permute(0, 2, 3, 1).contiguous()  # (frame_train, H, W, 2)
            est_dynamic = torch.view_as_complex(est_dynamic)  # (frame_train, H, W)

            cano_feat = self.CANO_net_enc(self.xy_coord.reshape(-1, 2).to(device)).float() * cano_enc_mask
            cano_feat = cano_feat.reshape(self.H, self.W, 1, cano_feat.shape[-1]).float()
            cano_feat = cano_feat.permute(2, 3, 0, 1)
            est_cano = self.CANO_net_dec(cano_feat).float()
            est_cano = torch.view_as_complex(
                        est_cano.permute(0, 2, 3, 1).contiguous())  # (H, W, 2)
            est_cano = est_cano.squeeze()

            recon = est_dynamic.squeeze().detach().cpu()
            recon = recon * self.img_mask.cpu()  # Apply mask to the reconstruction
            recon = recon / torch.abs(recon).max()

            psnr, ssim = cal_metrics(torch.abs(recon), 
                                     torch.abs(self.gt_img))
        
            print(f'PSNR/SSIM: {psnr:.2f}/{ssim:.3f}')

            error = torch.abs(torch.abs(recon.cpu()) - torch.abs(self.gt_img.cpu()))
            sitk.WriteImage(sitk.GetImageFromArray(error), f'{self.save_path}/dynamic/error_{epoch+1}.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(torch.abs(recon)), f'{self.save_path}/dynamic/recon_{epoch+1}.nii.gz')
            cano_est = est_cano.detach().cpu().numpy()
            sitk.WriteImage(sitk.GetImageFromArray(np.abs(cano_est)), f'{self.save_path}/cano/cano_{epoch+1}.nii.gz')

        return recon

        


    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save({
            'DVF_net_enc_state_dict': self.DVF_net_enc.state_dict(),
            'DVF_net_dec_state_dict': self.DVF_net_dec.state_dict(),
            'CANO_net_enc_state_dict': self.CANO_net_enc.state_dict(),
            'CANO_net_dec_state_dict': self.CANO_net_dec.state_dict(),
        }, path + 'model.pth')
