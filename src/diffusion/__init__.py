from .paper_diffusion import (
    VESDEDiffusion,
    VPDDPMDiffusion,
    get_geometric_sigma_schedule,
    get_linear_beta_schedule,
)


def create_diffusion(mode: str = "ve_sde", timesteps: int = 1000):
    if mode == "ve_sde":
        sigmas = get_geometric_sigma_schedule(timesteps)
        return VESDEDiffusion(sigmas=sigmas)
    if mode == "vp_ddpm":
        betas = get_linear_beta_schedule(timesteps)
        return VPDDPMDiffusion(betas=betas)
    raise ValueError(f"Unknown diffusion mode: {mode}")