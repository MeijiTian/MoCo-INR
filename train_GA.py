import json
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
from scipy import io
from tqdm import tqdm
import torchkbnufft as tkbn

from utils import fftnc, ifftnc, cal_metrics, gen_traj
from src.loss import  L1Loss, DVFRegLoss
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
    ktraj: torch.Tensor,
    train_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    """
    If train_frame == 0: use all frames.
    Else: randomly select a contiguous frame window.
    """
    n_frame = kdata.shape[0]

    if train_frame == 0:
        return kdata, ktraj, None

    start = np.random.randint(0, n_frame - train_frame + 1)
    end = start + train_frame
    return kdata[start:end], ktraj[start:end], start


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # -------------------- config --------------------
    config_path = Path("Config/GA_recon.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = Path(config["data"]["data_path"])
    n_spoke = int(config["data"]["n_spoke"])
    kscale = float(config["train"]["kscale"])

    save_root = Path(config["evaluation"]["save_path"]) / f"n_spoke_{n_spoke}"
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
    spoke_length = grid_size * 2
    
    GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2)) 
    ktraj = gen_traj(GA, spoke_length, n_frame * n_spoke, ind=0).to(device)  # (2, n_spoke*grid_size)
    ktraj = ktraj.reshape(2, n_frame, -1).transpose(1, 0)


    print("GT image shape:", gt_img.shape)
    print("Smap shape:", smap.shape)
    print("Ktraj shape:", ktraj.shape)


    # -------------------- tensors --------------------
    gt = torch.from_numpy(gt_img).to(device)                 # (T, H, W)
    smap_t = torch.from_numpy(smap).to(device)               # (C, H, W)
    nufft_ob = tkbn.KbNufft(im_size=(grid_size, grid_size)).to(device)


    # k-space data (masked)
    # (T, 1, H, W) * (1, C, H, W) -> (T, C, H, W)
    coil_imgs = gt.unsqueeze(1) * smap_t.unsqueeze(0)

    kdata = nufft_ob(coil_imgs, ktraj, norm = 'ortho')  # (T, C, n_spoke*spoke_length)
    kdata = kdata * kscale

    # -------------------- ZF baseline --------------------

    adj_nufft_ob = tkbn.KbNufftAdjoint(im_size=(grid_size, grid_size)).to(torch.complex64).to(device)
    dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([n_spoke, 1]).to(device)
    ZF = adj_nufft_ob(kdata * dcomp.flatten(), 
                       ktraj, smaps=smap_t, norm = 'ortho')
    ZF = ZF.squeeze(1)  # (T, H, W)
    ZF = ZF / torch.abs(ZF).max().clamp_min(1e-12)
    psnr, ssim = cal_metrics(torch.abs(ZF), torch.abs(gt))
    save_nii(
        torch.abs(ZF).detach().cpu().numpy(),
        str(recon_dir / f"GA_NUFFT_recon.nii.gz"),
    )
    print(f"PSNR/SSIM for iNUFFT recon: {psnr:.2f}/{ssim:.3f}")

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
        target_k, ktraj_batch, start_frame = choose_frame_batch(kdata, ktraj, train_frame=train_frame)

        # forward
        out = model.train_step(epoch, start_frame=start_frame)
        est_recon = out["est_recon"]  # (T', 1, H, W) complex
        est_dvf = out["est_dvf"]      # (T', 2, H, W)

        # k-space of current estimate
        est_k = nufft_ob(est_recon * smap_t.unsqueeze(0), 
                         ktraj_batch, norm = 'ortho')  # (T', C, n_spoke*spoke_length)

        # DC loss on sampled locations only
        # mask_batch: (T', H, W) -> broadcast to (T', C, H, W)
        dc_loss = dc_loss_fn(est_k, target_k)

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
