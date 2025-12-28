import json
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
from scipy import io
from tqdm import tqdm

from utils import fftnc, ifftnc, cal_metrics
from src.loss import L1Loss, DVFRegLoss
from src.utils import save_nii, plot_dvf_frames
from src.model import MoCoINR




# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_nii_as_np(path: str | Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def build_sampling_mask(data_path: Path, grid_size: int, n_frame: int, af: int) -> np.ndarray:
    mat_path = data_path / f"samp_VISTA_{grid_size}x{n_frame}_R{af}.mat"
    samp = io.loadmat(str(mat_path))["samp"]  # expected shape: (grid_size, n_frame) or (n_frame, grid_size)
    samp = samp.T  # keep your original transposition
    samp = samp[:, :, None].repeat(grid_size, axis=2)  # (n_frame, H, W)
    return samp


def zero_filled_recon(
    kdata: torch.Tensor,
    smap: torch.Tensor,
) -> torch.Tensor:
    """
    kdata: (T, C, H, W) complex
    smap:  (C, H, W) complex
    return: (T, H, W) complex
    """
    zf = ifftnc(kdata, dim=(-2, -1))
    zf = torch.sum(zf * smap.unsqueeze(0).conj(), dim=1)
    zf = zf / torch.abs(zf).max().clamp_min(1e-12)
    return zf


def choose_frame_batch(
    kdata: torch.Tensor,
    mask: torch.Tensor,
    train_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    """
    If train_frame == 0: use all frames.
    Else: randomly select a contiguous frame window.
    """
    n_frame = kdata.shape[0]

    if train_frame == 0:
        return kdata, mask, None

    start = np.random.randint(0, n_frame - train_frame + 1)
    end = start + train_frame
    return kdata[start:end], mask[start:end], start


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # -------------------- config --------------------
    config_path = Path("Config/VISTA_recon.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = Path(config["data"]["data_path"])
    af = int(config["data"]["AF"])
    kscale = float(config["train"]["kscale"])

    save_root = Path(config["evaluation"]["save_path"]) / f"AF_{af}"
    config["evaluation"]["save_path"] = str(save_root) + "/"

    # directories
    recon_dir = ensure_dir(save_root / "Recon")
    cano_dir = ensure_dir(save_root / "Cano")
    dvf_dir = ensure_dir(save_root / "DVF")

    # -------------------- device --------------------
    gpu = config["train"]["gpu"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # -------------------- load data --------------------
    gt_img = load_nii_as_np(data_path / "gt_img.nii.gz")     # (T, H, W)
    smap = load_nii_as_np(data_path / "csm.nii.gz")          # (C, H, W)

    n_coil, H, W = smap.shape
    n_frame = gt_img.shape[0]
    grid_size = gt_img.shape[-1]  # keep your original naming

    samp_mask_np = build_sampling_mask(data_path, grid_size=grid_size, n_frame=n_frame, af=af)

    print("GT image shape:", gt_img.shape)
    print("Smap shape:", smap.shape)
    print("Sampling mask shape:", samp_mask_np.shape)

    # -------------------- tensors --------------------
    gt = torch.from_numpy(gt_img).to(device)                 # (T, H, W)
    smap_t = torch.from_numpy(smap).to(device)               # (C, H, W)
    mask = torch.from_numpy(samp_mask_np).to(device)         # (T, H, W) (0/1)

    # k-space data (masked)
    # (T, 1, H, W) * (1, C, H, W) -> (T, C, H, W)
    coil_imgs = gt.unsqueeze(1) * smap_t.unsqueeze(0)
    kdata = fftnc(coil_imgs, dim=(-2, -1)) * mask.unsqueeze(1)
    kdata = kdata * kscale

    # -------------------- ZF baseline --------------------
    zf = zero_filled_recon(kdata, smap_t)
    psnr, ssim = cal_metrics(torch.abs(zf), torch.abs(gt))
    print(f"PSNR/SSIM for ZF recon: {psnr:.2f}/{ssim:.3f}")

    # -------------------- model & optim --------------------
    model = MoCoINR(
        config,
        n_frame=n_frame,
        img_size=(grid_size, grid_size),
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    dc_loss_fn = L1Loss().to(device)
    dvf_reg_fn = DVFRegLoss(weight=1.0).to(device)

    # -------------------- training loop --------------------
    epochs = int(config["train"]["epoch"])
    train_frame = int(config["train"]["train_frame"])
    summary_epoch = int(config["evaluation"]["summary_epoch"])

    pbar = tqdm(range(epochs), colour="blue", ncols=100)

    for epoch in pbar:
        target_k, mask_batch, start_frame = choose_frame_batch(kdata, mask, train_frame=train_frame)

        # forward
        out = model.train_step(epoch, start_frame=start_frame)
        est_recon = out["est_recon"]  # (T', 1, H, W) complex
        est_dvf = out["est_dvf"]      # (T', 2, H, W)

        # k-space of current estimate
        est_k = fftnc(est_recon * smap_t.unsqueeze(0), dim=(-2, -1))  # (T', C, H, W)

        # DC loss on sampled locations only
        # mask_batch: (T', H, W) -> broadcast to (T', C, H, W)
        sampled = mask_batch.unsqueeze(1).bool().repeat(1, n_coil, 1, 1)
        dc_loss = dc_loss_fn(est_k[sampled==1], target_k[sampled==1])

        # regularize DVF
        reg_dvf = dvf_reg_fn(est_dvf)

        loss = dc_loss + reg_dvf
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"DC": f"{dc_loss.item():.4g}", "RegDVF": f"{reg_dvf.item():.4g}"})

        # -------------------- evaluation --------------------
        if (epoch + 1) % summary_epoch != 0:
            continue

        eval_out = model.evaluate(epoch)
        est_recon_full = eval_out["est_recon"]  # (T, H, W) complex
        est_cano = eval_out["est_cano"]         # (H, W) complex
        est_dvf_full = eval_out["est_dvf"]      # (T, 2, H, W)

        # DVF to (T, H, W, 2) for saving/plotting
        est_dvf_hw2 = est_dvf_full.permute(0, 2, 3, 1).contiguous()

        # apply ROI mask (keep your behavior)
        img_mask = torch.ones_like(gt)
        img_mask[torch.abs(gt) == 0] = 0
        est_recon_full = est_recon_full * img_mask

        # normalize
        est_recon_full = est_recon_full / torch.abs(est_recon_full).max().clamp_min(1e-12)

        psnr, ssim = cal_metrics(torch.abs(est_recon_full), torch.abs(gt))
        print(f"Epoch {epoch + 1}: PSNR/SSIM: {psnr:.2f}/{ssim:.3f}")

        # save recon
        save_nii(
            torch.abs(est_recon_full).detach().cpu().numpy(),
            str(recon_dir / f"MoCoINR_est_{epoch + 1}.nii.gz"),
        )

        # save cano
        est_cano_norm = est_cano / torch.abs(est_cano).max().clamp_min(1e-12)
        save_nii(
            torch.abs(est_cano_norm).detach().cpu().numpy(),
            str(cano_dir / f"MoCoINR_cano_{epoch + 1}.nii.gz"),
        )

        # save dvf
        save_nii(
            est_dvf_hw2.detach().cpu().numpy(),
            str(dvf_dir / f"MoCoINR_DVF_{epoch + 1}.nii.gz"),
        )

        plot_dvf_frames(
            est_dvf_hw2.detach().cpu().numpy(),
            frames=[0, n_frame // 4, n_frame // 2, 3 * n_frame // 4, n_frame - 1],
            interval=2,
            save_path=str(dvf_dir / f"MoCoINR_DVF_frames_{epoch + 1}.png"),
        )


if __name__ == "__main__":
    main()
