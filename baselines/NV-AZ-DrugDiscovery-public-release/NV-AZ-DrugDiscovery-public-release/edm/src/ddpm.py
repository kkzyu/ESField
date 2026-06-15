import numpy as np
import torch


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = torch.nn.Parameter(x, requires_grad=False)
    return x


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    # numerically stable version
    # of log( exp(a) + exp(b) )
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def consistify_shape(src, trg):
    new_shape = (-1,) + (1,) * (len(trg.shape) - 1)
    return src.view(*new_shape)


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]

    alphas = np.clip(alphas, a_min=0.001, a_max=1.0)

    # This corresponds to alpha_sqrt in Ho et al. paper
    # for the Gaussian diffusion
    alphas = np.sqrt(alphas)
    return alphas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def sample_time(batch_size, timesteps, method):
    if method == "symmetric":
        time_step = torch.randint(0, timesteps, size=(batch_size // 2 + 1,))
        time_step = torch.cat([time_step, timesteps - time_step - 1], dim=0)[
            :batch_size
        ]
        pt = torch.ones_like(time_step).float() / timesteps
        return time_step, pt

    else:
        raise ValueError


class ContinuosDiffusionSampler(torch.nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        self.timesteps = timesteps
        # self.beta = 0.008

        # betas = cosine_beta_schedule(self.timesteps, self.beta) ** 2
        betas = get_beta_schedule(
            "sigmoid",
            beta_start=1.0e-7,
            beta_end=2.0e-3,
            num_diffusion_timesteps=self.timesteps,
        )
        alphas = 1.0 - betas
        bar_alphas = np.cumprod(alphas, axis=0)
        bar_alphas_prev = np.append(1.0, bar_alphas[:-1])
        posterior_var = (1.0 - bar_alphas_prev) / (1.0 - bar_alphas) * betas

        # \bar{\alpha_t} = \prod_{s=1}^t \alpha_s
        self.bar_alphas = to_torch_const(bar_alphas)

        # \frac{ \sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t}
        self.posterior_mean_x_0_coef = to_torch_const(
            np.sqrt(bar_alphas_prev) * betas / (1.0 - bar_alphas)
        )

        # \frac{ \sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t}
        self.posterior_mean_x_t_coef = to_torch_const(
            np.sqrt(alphas) * (1.0 - bar_alphas_prev) / (1.0 - bar_alphas)
        )

        # \log frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_t
        self.log_posterior_var_coef = to_torch_const(
            np.log(np.append(posterior_var[1], posterior_var[1:]))
        )

    def sample_xt_given_x0(self, x_0, t, mask):
        r"""
        q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)

        \alpha_t = 1 - \beta_t

        \bar{\alpha}_t = \prod_{s=1}^t \alpha_s

        """
        inv_mask = (~mask).float()
        for _ in range(len(x_0.shape) - len(inv_mask.shape)):
            inv_mask = inv_mask.unsqueeze(-1)

        bar_alpha_t = self.bar_alphas[t]
        for _ in x_0.shape[1:]:
            bar_alpha_t = bar_alpha_t.unsqueeze(-1)

        noise = torch.randn_like(x_0)
        # reparametrization trick
        # for sampling q(x_t | x_0)
        x_t = torch.sqrt(bar_alpha_t) * x_0 + torch.sqrt(1.0 - bar_alpha_t) * noise
        return x_t * inv_mask

    def add_noise(self, x, t, mask):
        """
        Wrapper around self.sample_xt_given_x0
        """
        x_t = self.sample_xt_given_x0(x, t, mask)
        return x_t

    def q_posterior(self, x_0, x_t, t, mask):
        r"""
        q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)

        \tilde{\mu}_t(x_t, x_0) = \frac{ \sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 +
                                  \frac{ \sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t

        \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_t

        """
        inv_mask = (~mask).float()
        for _ in range(len(x_0.shape) - len(inv_mask.shape)):
            inv_mask = inv_mask.unsqueeze(-1)

        posterior_mean_x_0_coef = self.posterior_mean_x_0_coef[t]
        posterior_mean_x_t_coef = self.posterior_mean_x_t_coef[t]

        for _ in x_0.shape[1:]:
            posterior_mean_x_0_coef = posterior_mean_x_0_coef.unsqueeze(-1)
            posterior_mean_x_t_coef = posterior_mean_x_t_coef.unsqueeze(-1)

        x_t_minus_1 = posterior_mean_x_0_coef * x_0 + posterior_mean_x_t_coef * x_t
        return x_t_minus_1 * inv_mask

    def get_loss(self, x_0, x_0_hat, mask):
        """
        This is a way to model the denoiser step of diffusion process
        through a neural network f.
        x_0_hat = f(x_t, t); x_0_hat = f(x_{t-1}, t-1); ...
        """
        dims = len(x_0.shape)
        inv_mask = (~mask).float()
        for _ in range(dims - len(inv_mask.shape)):
            inv_mask = inv_mask.unsqueeze(-1)

        span_dims = tuple(range(1, dims))
        loss = ((x_0_hat * inv_mask - x_0) ** 2).sum(span_dims) / inv_mask.sum(
            span_dims
        )
        loss = loss.mean()
        return loss

    def get_current_device(self):
        return self.bar_alphas.device

    @torch.no_grad()
    def predict_eps_from_x0_hat(self, x_0_hat_pos, x_t_pos, t, mask=None):
        sqrt_recip_alphas_cumprod = torch.sqrt(self.bar_alphas[t])
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1 - self.bar_alphas[t])
        for _ in x_0_hat_pos.shape[1:]:
            sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.unsqueeze(-1)
            sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod.unsqueeze(-1)
        num = x_t_pos - x_0_hat_pos * sqrt_recip_alphas_cumprod
        return num / sqrt_recipm1_alphas_cumprod

    @torch.no_grad()
    def predict_x0_hat_from_eps(self, eps_pos, x_t_pos, t, mask=None):
        sqrt_recip_alphas_cumprod = torch.sqrt(self.bar_alphas[t])
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1 - self.bar_alphas[t])
        for _ in eps_pos.shape[1:]:
            sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.unsqueeze(-1)
            sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod.unsqueeze(-1)

        return (
            x_t_pos - eps_pos * sqrt_recipm1_alphas_cumprod
        ) / sqrt_recip_alphas_cumprod

    @torch.no_grad()
    def update_eps_pos(self, eps_pos, guidance, t, beta):
        sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.bar_alphas[t])
        for _ in eps_pos.shape[1:]:
            sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.unsqueeze(-1)
        new_eps_pos = eps_pos - beta * sqrt_recip_alphas_cumprod * guidance
        return new_eps_pos

    @torch.no_grad()
    def sample_diffusion_step(self, x_0_hat, x_t, t, mask, deterministic=False):
        x_t_minus_1 = self.q_posterior(x_0=x_0_hat, x_t=x_t, t=t, mask=mask)

        if deterministic:
            return x_t_minus_1

        inv_mask = (~mask).float()
        for _ in range(len(x_0_hat.shape) - len(inv_mask.shape)):
            inv_mask = inv_mask.unsqueeze(-1)

        nonzero_t_mask = 1.0 - (t == 0).float()
        log_posterior_var_coef = self.log_posterior_var_coef[t]
        for _ in x_0_hat.shape[1:]:
            nonzero_t_mask = nonzero_t_mask.unsqueeze(-1)
            log_posterior_var_coef = log_posterior_var_coef.unsqueeze(-1)

        x_t_minus_1 = x_t_minus_1 + nonzero_t_mask * torch.exp(
            0.5 * log_posterior_var_coef
        ) * torch.randn_like(x_t)
        return x_t_minus_1 * inv_mask


class MultiCategoricalDiffusionSampler(torch.nn.Module):
    def __init__(
        self,
        timesteps,
        categorical_dims=[],
        paddings=None,
        loss_type="vb_stochastic",
        parametrization="x_0",
    ):
        """DDPM for sets of categorical tuples.

        Parameters
        ----------
        timesteps : int
            Number of timesteps in the diffusion process
        categorical_dims: list[int]
            Number of categories for of each categorical variable
        loss_type : str
            Type of loss:
            - 'vb_all': placeholder for now
            - 'vb_stochastic': approximation
        parametrization : str
            Behavior of the denoiser:
            - 'x_0': if the denoiser f(x_t) produces x_0
            - 'direct': if the denoiser f(x_t) produces x_{t-1}
        """
        super().__init__()
        assert loss_type in ("vb_stochastic", "vb_all")
        assert parametrization in ("x_0", "direct")

        if loss_type == "vb_all":
            print(
                "Computing the loss using the bound on _all_ timesteps."
                " This is expensive both in terms of memory and computation."
            )
            raise NotImplementedError("vb_all not implemented")

        # Per-step transition probability for the multinomial categorical
        # diffusion kernel (see Hoogeboom et al., 2021 — Argmax Flows):
        # at each timestep the atom type has probability `beta` of being
        # resampled uniformly. The value 0.008 matches the original EDM
        # categorical schedule used during training.
        self.beta = 0.008
        self.timesteps = timesteps
        self.dims = categorical_dims
        self.loss_type = loss_type
        self.categorical_dims = categorical_dims
        if paddings is None or len(paddings) == 0:
            self.paddings = [0 for _ in self.categorical_dims]
        else:
            assert len(self.paddings) == len(self.categorical_dims)
            self.paddings = paddings
        self.parametrization = parametrization

        alphas = cosine_beta_schedule(self.timesteps, s=0.01)
        log_alphas = np.log(alphas)
        log_alphas_cumprod = np.cumsum(log_alphas)

        self.alphas = to_torch_const(alphas)
        self.log_alphas = to_torch_const(log_alphas)
        self.log_one_minus_alphas = to_torch_const(log_1_min_a(log_alphas))

        self.log_alphas_cumprod = to_torch_const(log_alphas_cumprod)
        self.log_one_minus_alphas_cumprod = to_torch_const(
            log_1_min_a(log_alphas_cumprod)
        )

        self.Lt_history = to_torch_const(np.zeros(self.timesteps, dtype=np.float32))
        self.Lt_count = to_torch_const(np.zeros(self.timesteps, dtype=np.float32))

    def q_xt_given_xt_minus_one(self, X_t_minus_1, t):
        """
        It computes q(x_t | x_{t-1}) = \alpha_t x_{t-1} + (1-\alpha_t)/K for categorical variables

        x_t_minus_1: list or torch.Tensor. Each element in the list has size (B,L,C) # log_softmax or logits
        t: time. torch.Tensor of size (B,)
        """
        t = t.ravel()
        assert isinstance(X_t_minus_1, (list, torch.Tensor))
        # q(x_t | x_{t-1})
        return_list = True
        if not isinstance(X_t_minus_1, list):
            X_t_minus_1 = [X_t_minus_1]
            return_list = False
        assert len(X_t_minus_1) == len(self.categorical_dims)
        X_t = []
        for K, x_t_minus_1 in zip(self.categorical_dims, X_t_minus_1):
            log_alpha_t = consistify_shape(self.log_alphas[t], x_t_minus_1)
            log_one_minus_alpha_t = consistify_shape(
                self.log_one_minus_alphas[t], x_t_minus_1
            )

            x_t = log_add_exp(
                x_t_minus_1 + log_alpha_t, log_one_minus_alpha_t - np.log(K)
            )
            X_t.append(x_t)

        if return_list:
            return X_t
        return X_t[0]

    def q_xt_given_x0(self, X_0, t):
        """
        It computes q(x_t | x_0) = \bar{\alpha}_t x_{t-1} + (1-\bar{\alpha}_t)/K for categorical variables

        X_0: list or torch.Tensor. Each element in the list has size (B,L,C) # log_softmax or logits
        t: time. torch.Tensor of size (B,)
        """
        t = t.ravel()
        assert isinstance(X_0, (list, torch.Tensor))

        return_list = True
        if not isinstance(X_0, list):
            X_0 = [X_0]
            return_list = False
        assert len(X_0) == len(self.categorical_dims)
        X_t = []
        for K, x_0 in zip(self.categorical_dims, X_0):
            log_alpha_t = consistify_shape(self.log_alphas_cumprod[t], x_0)
            log_one_minus_alpha_t = consistify_shape(
                self.log_one_minus_alphas_cumprod[t], x_0
            )

            x_t = log_add_exp(x_0 + log_alpha_t, log_one_minus_alpha_t - np.log(K))
            X_t.append(x_t)

        if return_list:
            return X_t
        return X_t[0]

    def q_posterior(self, X_t, X_0, t):
        r"""
        It computes q(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}, x_0) * q(x_{t-1} | x_0) / q(x_t | x_0) = \mathcal{C}(x_{t-1} | \tilde{c}_t(x_t, x_0)) for categorical variables

        X_t, X_0: list or torch.Tensor. Each element in the list has size (B,L,C)
        t: time. torch.Tensor of size (B,)
        """
        t = t.ravel()
        t_minus_one = t - 1
        # negative values are not used in the final decoder
        t_minus_one = torch.where(
            t_minus_one < 0, torch.zeros_like(t_minus_one), t_minus_one
        )

        assert isinstance(X_t, (list, torch.Tensor))
        assert isinstance(X_0, (list, torch.Tensor))

        return_list = True
        if not isinstance(X_0, list):
            X_0 = [X_0]
            return_list = False
        if not isinstance(X_t, list):
            X_t = [X_t]
            return_list = False
        assert len(X_t) == len(self.categorical_dims)
        assert len(X_0) == len(self.categorical_dims)

        # We need to compute c^*_t(x_t, x_0) = ( \alpha_t x_t + (1-\alpha_t)/K ) \odot ( \bar{\alpha}_{t-1} x_0 + (1-\bar{\alpha}_{t-1})/K )

        # \alpha_t x_t + (1-\alpha_t)/K
        C_start_right_term = self.q_xt_given_x0(X_0, t_minus_one)
        # Fix X_t_minus_1
        for i, x_0 in enumerate(X_0):
            t_broadcast = consistify_shape(t, x_0)
            C_start_right_term[i] = torch.where(
                t_broadcast == 0, x_0, C_start_right_term[i]
            )

        # \bar{\alpha}_{t-1} x_0 + (1-\bar{\alpha}_{t-1})/K
        C_start_left_term = self.q_xt_given_xt_minus_one(X_t, t)

        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        # X_t_minus_1_given_X_t_and_X_0  = \tilde{c}_t(x_t, x_0) = NORMALIZED c^*_t(x_t, x_0)
        X_t_minus_1_given_X_t_and_X_0 = []

        for c_start_left_term, c_start_right_term in zip(
            C_start_left_term, C_start_right_term
        ):
            c_star = c_start_left_term + c_start_right_term
            c_star = c_star - torch.logsumexp(c_star, dim=-1, keepdims=True)
            X_t_minus_1_given_X_t_and_X_0.append(c_star)

        if return_list:
            return X_t_minus_1_given_X_t_and_X_0
        return X_t_minus_1_given_X_t_and_X_0[0]

    def index_to_log_onehot(self, x, K=None):
        X = []
        if K is not None:
            x_onehot = torch.nn.functional.one_hot(x[..., 0], K)
            log_x = torch.log(x_onehot.float().clamp(min=1e-30))
            X.append(log_x)
        else:
            for i, K in enumerate(self.categorical_dims):
                x_onehot = torch.nn.functional.one_hot(x[..., i], K)
                log_x = torch.log(x_onehot.float().clamp(min=1e-30))
                X.append(log_x)
        return X

    def add_noise(self, x, t, mask):
        X_0 = self.index_to_log_onehot(x)  # log_all_x_start
        X_t = self.sample_xt_given_x0(X_0, t, mask)
        return X_0, X_t

    def sample_xt_given_x0(self, X_0, t, mask):
        X_t = self.q_xt_given_x0(X_0, t)
        samples = self.log_sample_categorical(X_t, mask)
        return samples

    def log_sample_categorical(self, X, mask):
        # mask has size B x MAX CARDINALITY
        # 0 to use / 1 to ignore
        samples = []
        for x, K, pad in zip(X, self.categorical_dims, self.paddings):
            uniform = torch.rand_like(x)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            # prevent to move categories to padding
            # gumbel_noise[..., pad] = -5. # lower than -torch.log(-torch.log(1e-30) + 1e-30)
            # sample = (gumbel_noise + x).argmax(dim=-1)

            sample = gumbel_noise + x
            sample[..., pad] = sample.detach().min() - 1.0
            sample = sample.argmax(dim=-1)
            sample[mask] = pad

            sample = self.index_to_log_onehot(sample[..., None], K)
            samples.append(sample[0])
        return samples

    def multinomial_kl(self, X, Y):
        kl = [(x.exp() * (x - y)).sum(dim=-1) for x, y in zip(X, Y)]
        return kl

    def kl_prior(self, X_0, mask):
        b = X_0[0].size(0)
        ones = torch.ones(b, device=self.get_current_device(), dtype=torch.long)

        X_t = self.q_xt_given_x0(X_0, t=(self.timesteps - 1) * ones)
        log_all_half_prob = [
            -torch.log(K * torch.ones_like(x_t))
            for x_t, K in zip(X_t, self.categorical_dims)
        ]

        kl_prior = self.multinomial_kl(X_t, log_all_half_prob)
        kl_prior = [
            (k * (~mask)).sum([i for i in range(1, len(k.shape))]) for k in kl_prior
        ]
        kl_prior = torch.stack(kl_prior, -1).sum(-1)
        return kl_prior

    def get_loss(self, X_0, X_t, X_hat, t, mask, t_weight=None):
        """
        x: tensor B x len(self.categorical_dims)
        X_hat list of len(self.categorical_dims) model predictions
        """

        t = t.ravel()
        if self.loss_type == "vb_stochastic":
            X_t_minus_1_given_X_t_and_X_0 = self.q_posterior(X_t, X_0, t)
            if self.parametrization == "x_0":
                X_hat_t_minus_1 = self.q_posterior(X_t, X_hat, t=t)
            elif self.parametrization == "direct":
                X_hat_t_minus_1 = X_hat

            kl = self.multinomial_kl(X_t_minus_1_given_X_t_and_X_0, X_hat_t_minus_1)

            model_nll = [
                -log_categorical(x_0, x_hat_t_minus_1)
                for x_0, x_hat_t_minus_1 in zip(X_0, X_hat_t_minus_1)
            ]

            kl = [(k * (~mask)).sum([i for i in range(1, len(k.shape))]) for k in kl]
            model_nll = [
                (d * (~mask)).sum([i for i in range(1, len(d.shape))])
                for d in model_nll
            ]

            kl, model_nll = torch.stack(kl, -1).sum(-1), torch.stack(model_nll, -1).sum(
                -1
            )

            t_mask = (t == torch.zeros_like(t)).float()
            kl = t_mask * model_nll + (1.0 - t_mask) * kl

            # This part it is use to sample t
            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(X_0, mask)  # regularization term
            if t_weight is None:
                t_weight = torch.ones_like(t).float() / self.timesteps
            # Upweigh loss term of the kl
            loss = kl / t_weight + kl_prior
            return loss.sum() / (np.log(2) * (mask == 0).sum().float())

        else:
            raise NotImplementedError

    def sample_time(self, b, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt
        elif method == "uniform":
            t = torch.randint(
                0, self.timesteps, (b,), device=self.get_current_device()
            ).long()
            pt = torch.ones_like(t).float() / self.timesteps
            return t, pt
        else:
            raise ValueError

    @torch.no_grad()
    def sample_diffusion_step(self, X_hat, x_t, t, mask):
        if not isinstance(X_hat, list):
            X_hat = [X_hat]
        X_t = self.index_to_log_onehot(x_t)
        X_t_minus_1 = self.q_posterior(X_t, X_hat, t)
        x_t_minus_1 = self.log_sample_categorical(X_t_minus_1, mask)
        return x_t_minus_1

    def get_current_device(self):
        return self.log_one_minus_alphas.device
