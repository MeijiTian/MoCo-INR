import torch
import torch.nn.functional as F



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
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() > 4:
            raise ValueError('Input tensor is more than 4D')

        x_grad = torch.mean(torch.abs(self._Dx(x)))
        y_grad = torch.mean(torch.abs(self._Dy(x)))

        return self.weight * (x_grad + y_grad)
    
    def _Dx(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[:, :, :-1, :] = x[:, :, 1:, :]
        y[:, :, -1, :] = x[:, :, 0, :]
        return y - x
    
    def _Dy(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[:, :, :, :-1] = x[:, :, :, 1:]
        y[:, :, :, -1] = x[:, :, :, 0]
        return y - x

class LaplacianLoss2d(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(LaplacianLoss2d, self).__init__()
        self.weight = weight
        self.kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # x shape (grid_size, grid_size, n_frame, 2)
        N, C, H, W = x.shape
        kernel = self.kernel.to(x.device).repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
        laplacian = F.conv2d(x, kernel, padding=1, groups=2)
        loss = torch.mean(torch.abs(laplacian))  # Mean absolute error

        return self.weight * loss
    
class DVFRegLoss(torch.nn.Module):
    """Regularization for DVF: TV + Laplacian + L1 magnitude."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.tv_loss = SpatialTV()
        self.lap_loss = LaplacianLoss2d()

    def forward(self, dvf: torch.Tensor) -> torch.Tensor:
        loss = self.tv_loss(dvf) + self.lap_loss(dvf) + 0.5 * torch.mean(torch.abs(dvf))
        return self.weight * loss
