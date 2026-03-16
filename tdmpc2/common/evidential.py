import torch

def aleatoric_uncertainty(nu, alpha, beta) -> torch.Tensor:
    return torch.sqrt((beta*(1+nu)) / (alpha*nu))

def epistemic_uncertainty(nu) -> torch.Tensor:
    return 1 / torch.sqrt(nu)
