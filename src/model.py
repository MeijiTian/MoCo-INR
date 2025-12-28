import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import tinycudann as tcnn


class MoCoINR(nn.Module):
    """
    MoCoINR:
      - DVF branch: (t, x, y) -> encoding -> conv decoder -> (dx, dy)
      - CANO branch: (x', y') -> encoding -> conv decoder -> complex image

    Notes:
      - Coordinates are normalized to [-0.5, 0.5]
      - C2F (coarse-to-fine) feature selection supports freezing earlier levels
    """

    def __init__(
        self,
        config: Dict,
        n_frame: int,
        img_size: Tuple[int, int],
        device: torch.device,
    ):
        super().__init__()

        self.device = device
        self.train_config = config["train"]
        self.net_config = config["net"]

        self.n_frame = n_frame
        self.H, self.W = img_size

        self._init_networks()
        self.xy_coord, self.txy_coord = self._init_train_coord()

    # -------------------------------------------------------------------------
    # Network init
    # -------------------------------------------------------------------------
    def _init_networks(self) -> None:
        # DVF: (t, x, y) -> feature -> conv decoder -> (dx, dy)
        self.DVF_net_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=self.net_config["dvf_net_enc"],
        ).to(self.device)

        self.DVF_net_dec = self._make_conv_decoder(
            in_ch=self.DVF_net_enc.n_output_dims,
            out_ch=2,
        ).to(self.device)

        # CANO: (x, y) -> feature -> conv decoder -> (real, imag)
        self.CANO_net_enc = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=self.net_config["cano_net_enc"],
        ).to(self.device)

        self.CANO_net_dec = self._make_conv_decoder(
            in_ch=self.CANO_net_enc.n_output_dims,
            out_ch=2,
        ).to(self.device)

    @staticmethod
    def _make_conv_decoder(in_ch: int, out_ch: int) -> nn.Module:
        """A small conv decoder used for both DVF and CANO branches."""
        return nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
        )

    # -------------------------------------------------------------------------
    # Coordinate init
    # -------------------------------------------------------------------------
    def _init_train_coord(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize:
          - xy_coord: (H, W, 2)
          - txy_coord: (n_frame, H, W, 3) where t is taken from x-axis subset
        """
        x = torch.linspace(-0.5, 0.5, self.H, device=self.device)
        y = torch.linspace(-0.5, 0.5, self.W, device=self.device)

        # t sampled by cropping x-axis to match n_frame (keep your original design)
        start = (self.H - self.n_frame) // 2
        end = (self.H + self.n_frame) // 2
        t = x[start:end]

        xx, yy = torch.meshgrid(x, y, indexing="ij")
        xy_coord = torch.stack([xx, yy], dim=-1)  # (H, W, 2)

        tt, xx3, yy3 = torch.meshgrid(t, x, y, indexing="ij")
        txy_coord = torch.stack([tt, xx3, yy3], dim=-1)  # (T, H, W, 3)

        # 如果你不想训练时打印，建议换成 logger 或删掉
        # print("txy_coord:", txy_coord.shape, "xy_coord:", xy_coord.shape)

        return xy_coord, txy_coord

    # -------------------------------------------------------------------------
    # C2F feature selection
    # -------------------------------------------------------------------------
    @staticmethod
    def _c2f_select_features(
        feat: torch.Tensor,
        level_frozen: int,
        level_train: int,
        enc_config: Dict,
    ) -> torch.Tensor:
        """
        Select features with C2F strategy:
          - levels [0, level_frozen) are detached (frozen)
          - levels [level_frozen, level_train) are trainable
          - levels [level_train, end) are masked out (kept 0)

        feat shape: (N, feat_dim)
        """
        n_feat_per_level = enc_config["n_features_per_level"]
        feat_dim = feat.shape[-1]

        frozen_end = level_frozen * n_feat_per_level
        train_end = level_train * n_feat_per_level
        frozen_end = min(frozen_end, feat_dim)
        train_end = min(train_end, feat_dim)

        out = torch.zeros_like(feat)
        if frozen_end > 0:
            out[:, :frozen_end] = feat[:, :frozen_end].detach()
        if train_end > frozen_end:
            out[:, frozen_end:train_end] = feat[:, frozen_end:train_end]
        return out

    def _apply_c2f_if_needed(
        self,
        feat: torch.Tensor,
        epoch: int,
        branch: str,  # "dvf" or "cano"
    ) -> torch.Tensor:
        """Table-driven C2F rules to remove duplicated if-else blocks."""
        if not self.train_config.get("C2F", False):
            return feat

        c2f_cfg = self.train_config["C2F_config"]

        if branch == "dvf":
            enc_cfg = self.net_config["dvf_net_enc"]
            rules = [
                (c2f_cfg["stage_1"], 0, 6),
                (c2f_cfg["stage_2"], 4, 8),
                (10**18, 6, 8),  # default last stage
            ]
        elif branch == "cano":
            enc_cfg = self.net_config["cano_net_enc"]
            rules = [
                (c2f_cfg["stage_1"], 0, 8),
                (c2f_cfg["stage_2"], 6, 10),
                (10**18, 8, 12),
            ]
        else:
            raise ValueError(f"Unknown branch: {branch}")

        for upper_epoch, level_frozen, level_train in rules:
            if epoch <= upper_epoch:
                return self._c2f_select_features(
                    feat,
                    level_frozen=level_frozen,
                    level_train=level_train,
                    enc_config=enc_cfg,
                )

        return feat  # should never reach

    # -------------------------------------------------------------------------
    # Helpers: reshape conventions
    # -------------------------------------------------------------------------
    @staticmethod
    def _flat_to_nchw(feat_flat: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Convert (N*H*W, C) -> (N, C, H, W) with N inferred.
        """
        c = feat_flat.shape[-1]
        feat = feat_flat.view(-1, H, W, c).permute(0, 3, 1, 2).contiguous()
        return feat

    # -------------------------------------------------------------------------
    # Forward-like API: train step / evaluate
    # -------------------------------------------------------------------------
    def train_step(self, epoch: int, start_frame: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        One training forward:
          return:
            - est_recon: (n_frame, 1, H, W) complex (stored as complex tensor)
            - est_dvf:   (n_frame, 2, H, W)
        """
        # Select frames
        if start_frame is None:
            txy_batch = self.txy_coord  # (T, H, W, 3)
        else:
            n_train = self.train_config["train_frame"]
            end_frame = start_frame + n_train
            txy_batch = self.txy_coord[start_frame:end_frame]  # (T', H, W, 3)

        # ---------------- DVF branch ----------------
        dvf_in = txy_batch.reshape(-1, 3)
        dvf_feat = self.DVF_net_enc(dvf_in).float()
        dvf_feat = self._apply_c2f_if_needed(dvf_feat, epoch=epoch, branch="dvf")

        dvf_feat_nchw = self._flat_to_nchw(dvf_feat, self.H, self.W)  # (T, C, H, W)
        xy_delta = self.DVF_net_dec(dvf_feat_nchw).float()            # (T, 2, H, W)
        xy_delta_hw2 = xy_delta.permute(0, 2, 3, 1).contiguous()      # (T, H, W, 2)

        # deform grid: (T,H,W,2)
        deform_grid = txy_batch[..., 1:3] + xy_delta_hw2

        # ---------------- CANO branch ----------------
        cano_in = deform_grid.reshape(-1, 2)
        cano_feat = self.CANO_net_enc(cano_in).float()
        cano_feat = self._apply_c2f_if_needed(cano_feat, epoch=epoch, branch="cano")

        cano_feat_nchw = self._flat_to_nchw(cano_feat, self.H, self.W)
        est_ri = self.CANO_net_dec(cano_feat_nchw).float()            # (T, 2, H, W)

        # complex: (T,H,W)
        est_dynamic = est_ri[:, 0] + 1j * est_ri[:, 1]

        return {
            "est_recon": est_dynamic.unsqueeze(1),  # (T,1,H,W)
            "est_dvf": xy_delta,                   # (T,2,H,W)
        }

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, torch.Tensor]:
        """
        Evaluate current model:
          - est_recon: (T, H, W) complex
          - est_cano:  (H, W) complex
          - est_dvf:   (T, 2, H, W)
        """
        out = self.train_step(epoch)
        est_recon = out["est_recon"].squeeze(1)  # (T,H,W)
        est_dvf = out["est_dvf"]                 # (T,2,H,W)

        # Canonical image from xy grid (single frame)
        cano_feat = self.CANO_net_enc(self.xy_coord.reshape(-1, 2)).float()
        cano_feat = self._apply_c2f_if_needed(cano_feat, epoch=epoch, branch="cano")
        cano_feat_nchw = self._flat_to_nchw(cano_feat, self.H, self.W)

        est_cano_ri = self.CANO_net_dec(cano_feat_nchw).float()  # (1,2,H,W) since N inferred is 1
        est_cano = torch.view_as_complex(
            est_cano_ri.permute(0, 2, 3, 1).contiguous()
        ).squeeze(0)  # (H,W)

        return {
            "est_recon": est_recon.detach(),
            "est_cano": est_cano.detach(),
            "est_dvf": est_dvf.detach(),
        }

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        ckpt = {
            "DVF_net_enc_state_dict": self.DVF_net_enc.state_dict(),
            "DVF_net_dec_state_dict": self.DVF_net_dec.state_dict(),
            "CANO_net_enc_state_dict": self.CANO_net_enc.state_dict(),
            "CANO_net_dec_state_dict": self.CANO_net_dec.state_dict(),
        }
        torch.save(ckpt, os.path.join(path, "model.pth"))
