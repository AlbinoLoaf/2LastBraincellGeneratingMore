import math
import numpy as np
import torch as th
import torch.nn as nn


def get_linear_beta_schedule(num_diffusion_timesteps: int = 1000) -> np.ndarray:
    """Linear schedule from Ho et al., matching the active path in get_named_beta_schedule."""
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)


def _extract_into_tensor(arr: np.ndarray, timesteps: th.Tensor, broadcast_shape: tuple) -> th.Tensor:
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)

def get_geometric_sigma_schedule(
    num_diffusion_timesteps: int = 1000,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
) -> np.ndarray:
    """
    Geometric noise schedule sigma_t from Song et al., bounded by sigma_max = 1.0
    so that p_T(x) ~ N(0, I) as stated in Sections 3.3-3.4 of the paper.
    """
    return np.geomspace(sigma_min, sigma_max, num_diffusion_timesteps, dtype=np.float64)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = 0.5 * (1.0 + th.erf(plus_in / math.sqrt(2.0)))
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = 0.5 * (1.0 + th.erf(min_in / math.sqrt(2.0)))
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs


class VPDDPMDiffusion(nn.Module):
    """
    Variance preserving Denoising Defusion Probabilistic Model
    cleanup of the active VP-DDPM path in GaussianDiffusion.
    Preserves the original schedule calculations and variable names
    while removing unused enums, DDIM branches, and nested wrapper calls.
    """

    def __init__(self, betas: np.ndarray):
        super().__init__()

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
            ) if len(self.posterior_variance) > 1 else np.array([])


        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start: th.Tensor, t: th.Tensor, noise: th.Tensor = None) -> th.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start: th.Tensor, x_t: th.Tensor, t: th.Tensor):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_log_variance(self, model_var_values: th.Tensor, x_shape: tuple, t: th.Tensor) -> th.Tensor:
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x_shape)
        frac = (model_var_values + 1) / 2
        return frac * max_log + (1 - frac) * min_log

    def training_loss(self, model: nn.Module, x_start: th.Tensor, y: th.Tensor) -> th.Tensor:
        # Functianally the same just flattened
        t = th.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        model_output = model(x_t, t, y)
        C = x_t.shape[1]
        eps_pred, model_var_values = th.split(model_output, C, dim=1)

        # MSE loss on epsilon prediction
        mse = ((noise - eps_pred) ** 2).mean(dim=list(range(1, len(x_start.shape))))

        # Variational lower-bound loss on learned variance (with mean detached)
        pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps_pred.detach())
        pred_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        pred_log_var = self._predict_log_variance(model_var_values, x_t.shape, t)

        true_mean, _, true_log_var_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        kl = normal_kl(true_mean, true_log_var_clipped, pred_mean, pred_log_var)
        kl = kl.mean(dim=list(range(1, len(x_start.shape)))) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=pred_mean, log_scales=0.5 * pred_log_var
        )
        decoder_nll = decoder_nll.mean(dim=list(range(1, len(x_start.shape)))) / np.log(2.0)

        vb = th.where((t == 0), decoder_nll, kl) * (self.num_timesteps / 1000.0)
        return (mse + vb).mean()

    @th.no_grad()
    def sample_with_cfg(
        self, model: nn.Module, labels: th.Tensor, num_classes: int, shape: tuple, cfg_scale: float = 4.0
    ) -> th.Tensor:
        device = labels.device
        n = labels.shape[0]
        img = th.randn(n, *shape, device=device)

        y_null = th.full((n,), num_classes, dtype=labels.dtype, device=device)
        y_cfg = th.cat([labels, y_null], dim=0)

        for i in reversed(range(self.num_timesteps)):
            t = th.tensor([i] * (2 * n), device=device)
            model_out = model.forward_with_cfg(th.cat([img, img], dim=0), t, y_cfg, cfg_scale)
            eps_pred, model_var_values = th.split(model_out, shape[0], dim=1)
            eps_pred, model_var_values = eps_pred[:n], model_var_values[:n]

            t_half = t[:n]
            pred_xstart = self._predict_xstart_from_eps(x_t=img, t=t_half, eps=eps_pred)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=img, t=t_half)
            model_log_var = self._predict_log_variance(model_var_values, img.shape, t_half)

            nonzero_mask = (t_half != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            img = model_mean + nonzero_mask * th.exp(0.5 * model_log_var) * th.randn_like(img)

        return img * 100.0


class VESDEDiffusion(nn.Module):
    """
    Discretized Variance-Exploding SDE matching Paper Sections 3.3-3.4,
    structured identically to VPDDPMDiffusion for direct side-by-side comparison.
    """

    def __init__(self, sigmas: np.ndarray, weight_by_sigma_sq: bool = True):
        super().__init__()
        self.weight_by_sigma_sq = weight_by_sigma_sq

        # Use float64 for accuracy, matching VPDDPMDiffusion.__init__
        sigmas = np.array(sigmas, dtype=np.float64)
        self.sigmas = sigmas
        assert len(sigmas.shape) == 1, "sigmas must be 1-D"
        assert (sigmas > 0).all(), "sigmas must be positive"

        self.num_timesteps = int(sigmas.shape[0])

        # Previous noise levels (sigma_{-1} = 0 at clean signal x_0)
        self.sigmas_prev = np.append(0.0, self.sigmas[:-1])
        assert self.sigmas_prev.shape == (self.num_timesteps,)

        # Discrete reverse-SDE coefficients for f(t) = 0, g(t)^2 dt = sigma_t^2 - sigma_{t-1}^2
        self.sigma_sq_diff = self.sigmas ** 2 - self.sigmas_prev ** 2
        self.sqrt_sigma_sq_diff = np.sqrt(self.sigma_sq_diff)
        self.recip_sigmas = 1.0 / self.sigmas

    def q_sample(self, x_start: th.Tensor, t: th.Tensor, noise: th.Tensor = None) -> th.Tensor:
        """
        Forward perturbation kernel from Paper Eq. (3):
            x_t = x_0 + sigma_t * epsilon
        """
        if noise is None:
            noise = th.randn_like(x_start)
        return x_start + _extract_into_tensor(self.sigmas, t, x_start.shape) * noise

    def training_loss(self, model: nn.Module, x_start: th.Tensor, y: th.Tensor) -> th.Tensor:
        """
        Denoising score matching loss from Paper Eq. (3):
            L = E[ || s_theta(x_t, t, y) + epsilon / sigma_t ||_2^2 ]
        """
        t = th.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        score_pred = model(x_t, t, y)

        if self.weight_by_sigma_sq:
            # Standard Song et al. lambda(t) = sigma_t^2 weighting to prevent 1/sigma_t explosion
            sigma_t = _extract_into_tensor(self.sigmas, t, x_start.shape)
            loss = ((sigma_t * score_pred + noise) ** 2).mean(dim=list(range(1, len(x_start.shape))))
        else:
            # Unweighted literal Paper Eq. (3)
            target_score = -noise * _extract_into_tensor(self.recip_sigmas, t, x_start.shape)
            loss = ((score_pred - target_score) ** 2).mean(dim=list(range(1, len(x_start.shape))))

        return loss.mean()

    @th.no_grad()
    def sample_with_cfg(
        self, model: nn.Module, labels: th.Tensor, num_classes: int, shape: tuple, cfg_scale: float = 4.0
    ) -> th.Tensor:
        """
        Reverse-time VE-SDE solver from Cui Paper Eq. (2) and Eq. (4):
            dx_t = -g(t)^2 * s_theta^guided(x_t, t, y) dt + g(t) dw_bar
        """
        device = labels.device
        n = labels.shape[0]
        img = th.randn(n, *shape, device=device) * float(self.sigmas[-1])

        y_null = th.full((n,), num_classes, dtype=labels.dtype, device=device)
        y_cfg = th.cat([labels, y_null], dim=0)

        for i in reversed(range(self.num_timesteps)):
            t = th.tensor([i] * (2 * n), device=device)
            score_pred = model.forward_with_cfg(th.cat([img, img], dim=0), t, y_cfg, cfg_scale)[:n]

            t_half = t[:n]
            drift = _extract_into_tensor(self.sigma_sq_diff, t_half, img.shape) * score_pred
            nonzero_mask = (t_half != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            diffusion = (
                nonzero_mask
                * _extract_into_tensor(self.sqrt_sigma_sq_diff, t_half, img.shape)
                * th.randn_like(img)
            )
            img = img + drift + diffusion

        return img * 100.0