import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from models.util import lt_log_determinant
from torch import triangular_solve
import gpytorch
from sklearn.decomposition import PCA
from models.util import gaussian_expected_log_lik_diag

torch.set_default_dtype(torch.double)

# Model
class SDE_GPSSM(nn.Module):
    def __init__(self, Y, L_over_2, D, device="cpu", X_init=None):
        super().__init__()
        self.device = device
        self.Y = Y
        self.N, self.M = Y.shape
        self.D = D
        self.Q = 2
        self.num_samplept = L_over_2  # L/2
        self.total_num_sample = self.num_samplept * self.Q
        self.log_noise = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)  # leaf node

        # 参数初始化
        self.log_weight = nn.Parameter(torch.randn(self.Q, 1, device=self.device), requires_grad=True)

        if self.Q == 1:
            self.mu = nn.Parameter(torch.zeros(self.Q, self.D, device=self.device), requires_grad=False)
        else:
            self.mu = nn.Parameter(torch.zeros(self.Q, self.D, device=self.device), requires_grad=True)

        self.log_std = nn.Parameter(torch.randn(self.Q, self.D, device=self.device), requires_grad=True)

        pca = PCA(n_components=self.D)
        self.pca_X = pca.fit_transform(self.Y)

        # X = torch.randn(self.N, self.D, device=self.device)

        # self.mu_x = nn.Parameter(torch.tensor(X, device=self.device), requires_grad=True)
        # self.log_sigma_x = nn.Parameter(torch.zeros(self.N, self.D, device=self.device), requires_grad=True)

        # 学习变分参数的网络
        self.pesudo_H = nn.Parameter(torch.randn(self.M, self.D))  # Linear map
        hidden_dim = 32
        self.shared_feature = nn.Sequential(
            nn.Linear(self.M, hidden_dim),
            nn.ReLU()
        )
        self.head_Ypseudo = nn.Linear(hidden_dim, self.M)
        self.head_logR = nn.Linear(hidden_dim, self.M)

        # or
        # self.pesudo_Y = nn.Parameter(torch.randn_like(Y))  # pseudo observation to be learned
        # self.pesudo_log_diag_R = nn.Parameter(torch.zeros(self.M))  # noise std

        # 初始化 SDE 的参数
        # Drift matrix A: (D x D)
        self.A = nn.Parameter(torch.randn(self.D, self.D) * 0.1, requires_grad=True)

        # Diffusion matrix B: (D x D') — use identity if D' == D and want isotropic noise
        self.B = nn.Parameter(torch.randn(self.D, self.D) * 0.1, requires_grad=True)


    ################## Likelihood Term ######################
    def _compute_sm_basis(self, mean_x, cov_x, x_star=None, f_eval=False):
        multiple_Phi = []
        current_sampled_spectral_list = []

        if f_eval:  # use to evaluate the latent function 'f'
            x = mean_x
        else:
            # std = F.softplus(cov_x)   # shape: N * Q
            # eps = torch.randn_like(std)          # don't preselect/prefix it in __init__ function
            # x = mean_x + eps * std
            # mean_x: (N, D)
            # cov_x: (N, D, D)
            # L = torch.linalg.cholesky(cov_x)  # (N, D, D)
            # eps = torch.randn(mean_x.shape, device=mean_x.device)  # (N, D)
            # x = mean_x + torch.einsum("nij,nj->ni", L, eps)  # (N, D)
            #
            x = mean_x

        SM_weight = F.softplus(self.log_weight)
        SM_std = F.softplus(self.log_std)

        for i_th in range(self.Q):  
            SM_eps = torch.randn(self.num_samplept, self.D, device=self.device)
            sampled_spectral_pt = self.mu[i_th] + SM_std[i_th] * SM_eps  # L/2 * Q

            if x_star is not None:
                current_sampled_spectral_list.append(sampled_spectral_pt)

            x_spectral = (2 * np.pi) * x.matmul(sampled_spectral_pt.t())    # N * L/2

            Phi_i_th = (SM_weight[i_th] / self.num_samplept).sqrt() * torch.cat([x_spectral.cos(), x_spectral.sin()], 1)

            multiple_Phi.append(Phi_i_th)

        if x_star is None:
            return torch.cat(multiple_Phi, 1)  #  N * (m * L）

        else:
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list):
                xstar_spectral = (2 * np.pi) * x_star.matmul(current_sampled.t())

                Phistar_i_th = (SM_weight[i_th] / self.num_samplept).sqrt() * torch.cat([xstar_spectral.cos(), xstar_spectral.sin()], 1)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)  #  N * (m * L),  N_star * (M * L)

    def _compute_gram_approximate(self, Phi):
        zitter = 1e-8
        noise = self.log_noise.exp()
        return Phi.t() @ Phi + (noise + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()

    def _compute_gram_approximate_2(self, Phi):  # shape:  N x N
        return Phi @ Phi.T

    def neg_log_likelihood(self, batch_y, mean_x, cov_x):
        obs_dim = batch_y.shape[1]
        obs_num = batch_y.shape[0]
        Phi = self._compute_sm_basis(mean_x, cov_x)
        noise_std = self.log_noise.exp().sqrt()
        noise = self.log_noise.exp()

        # negative log-marginal likelihood
        if Phi.shape[0] > Phi.shape[1]:  # if N > (m*L)
            Approximate_gram = self._compute_gram_approximate(Phi)  # shape:  (m * L） x  (m * L）
            L = torch.cholesky(Approximate_gram)
            Lt_inv_Phi_y = triangular_solve((Phi.t()).matmul(batch_y), L, upper=False)[0]

            neg_log_likelihood = (0.5 / noise) * (batch_y.pow(2).sum() - Lt_inv_Phi_y.pow(2).sum())
            neg_log_likelihood += lt_log_determinant(L)
            neg_log_likelihood += (-self.total_num_sample) * 2 * noise_std
            neg_log_likelihood += 0.5 * obs_num * (np.log(2 * np.pi) + 2 * noise_std)

        else:
            k_matrix = self._compute_gram_approximate_2(Phi=Phi)  # shape: N x N
            C_matrix = k_matrix + noise * torch.eye(self.N, device=self.device)
            L = torch.cholesky(C_matrix)  # shape: N x N
            L_inv_y = triangular_solve(batch_y, L, upper=False)[0]
            # compute log-likelihood by ourselves
            constant_term = 0.5 * obs_num * np.log(2 * np.pi) * obs_dim
            log_det_term = torch.diagonal(L, dim1=-2, dim2=-1).sum().log() * obs_dim
            yy = 0.5 * L_inv_y.pow(2).sum()
            neg_log_likelihood = (constant_term + log_det_term + yy).div(obs_dim * obs_num)

        return neg_log_likelihood

    ############## 计算 Posterior of X 需要的 term
    def update_posterior(self, Y, dt):
        # input: Y, new variational parameters
        # return: smooth_mean, smooth_cov, pseudo_y, pseudo_h, psedo_var, log_lik_pseudo (term 3)

        # pseudo_y (N, M)
        # pseudo_R (N, M)
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik(Y)

        log_lik_pseudo, (filter_mean, filter_cov) = self.filter(dt, pseudo_y, pseudo_var, self.pesudo_H)
        dt = np.concatenate([dt[1:], np.array([0.0])], axis=0) # 为什么要这个步骤
        smoother_mean, smoother_cov, _ = self.smoother(dt, filter_mean, filter_cov)

        return smoother_mean, smoother_cov, pseudo_y, pseudo_var, self.pesudo_H, log_lik_pseudo

    def filter(self, dt, pseudo_y, pseudo_var, pseudo_H):
        """
        Kalman filter on linear SDE:
            dx = A x dt + B dW
            y_n = H x(t_n) + r_n,   r_n ~ N(0, diag(pseudo_var))

        Inputs:
            dt         : (N,) time intervals
            pseudo_y   : (N, M)
            pseudo_var : (N, M)
            pseudo_H   : (M, D)
        Returns:
            log_lik_sum: scalar log-likelihood
            (filter_means, filter_covs): each (N, D), (N, D, D)
        """
        N, M = pseudo_y.shape
        D = self.D
        A = self.A
        B = self.B
        Q = B @ B.T
        H = pseudo_H

        m = torch.tensor(self.pca_X[0])
        P = torch.eye(D, device=self.device) * 0.1
        filter_means = []
        filter_covs = []
        log_liks = []

        for n in range(N):
            if n > 0:
                F = torch.linalg.matrix_exp(A * dt[n])  # (D, D)
                m = F @ m
                P = F @ P @ F.T + Q * dt[n]

            Rn = torch.diag(pseudo_var[n])
            S = H @ P @ H.T + Rn
            K = P @ H.T @ torch.linalg.inv(S)
            y_n = pseudo_y[n]
            m = m + K @ (y_n - H @ m)
            P = (torch.eye(D, device=self.device) - K @ H) @ P

            log_lik = torch.distributions.MultivariateNormal(H @ m, S).log_prob(y_n)
            filter_means.append(m)
            filter_covs.append(P)
            log_liks.append(log_lik)

        filter_means = torch.stack(filter_means)  # (N, D)
        filter_covs = torch.stack(filter_covs)  # (N, D, D)
        log_lik_sum = torch.stack(log_liks).sum()  # scalar

        return log_lik_sum, (filter_means, filter_covs)

    def smoother(self, dt, filter_means, filter_covs):
        """
        RTS smoother for linear SDE
        Inputs:
            dt            : (N,) time intervals
            filter_means  : (N, D)
            filter_covs   : (N, D, D)
        Returns:
            smooth_means  : (N, D)
            smooth_covs   : (N, D, D)
            smooth_gains  : (N-1, D, D)
        """
        A = self.A
        B = self.B
        Q = B @ B.T
        N, D = filter_means.shape

        smooth_means = [None] * N
        smooth_covs = [None] * N
        smooth_means[-1] = filter_means[-1]
        smooth_covs[-1] = filter_covs[-1]
        gains = []

        for n in reversed(range(N - 1)):
            F = torch.linalg.matrix_exp(A * dt[n])  # (D, D)
            P_pred = F @ filter_covs[n] @ F.T + Q * dt[n]

            # Smoothing gain
            G = filter_covs[n] @ F.T @ torch.linalg.inv(P_pred)

            m_smooth = filter_means[n] + G @ (smooth_means[n + 1] - F @ filter_means[n])
            P_smooth = filter_covs[n] + G @ (smooth_covs[n + 1] - P_pred) @ G.T

            smooth_means[n] = m_smooth
            smooth_covs[n] = P_smooth
            gains.append(G)

        smooth_means = torch.stack(smooth_means)  # (N, D)
        smooth_covs = torch.stack(smooth_covs)  # (N, D, D)
        gains = torch.stack(gains[::-1])  # (N-1, D, D)

        return smooth_means, smooth_covs, gains

    def compute_full_pseudo_lik(self, Y):
        # 用来生成 pseudo_y, pseudo_var, H 我们可以直接优化
        hidden = self.shared_feature(Y)

        # # natural parameterisation
        #
        # lambda1 = self.hidden_to_mu(hidden)
        # lambda2 = self.hidden_to_var(hidden)
        # lambda1, lambda2 = np.expand_dims(lambda2, axis=-1), np.expand_dims(lambda2, axis=-1)
        # lambda2 = F.softplus(lambda2)
        # # learn lambda2 <- 1 / lambda2
        # mean = lambda1 * lambda2
        # var = -0.5 * lambda2

        pseudo_y = self.head_Ypseudo(hidden)
        pseudo_var = F.softplus(self.head_logR(hidden))

        return pseudo_y, pseudo_var

    def expected_log_lik_diag(self, pseudo_y, mean_x, cov_x, pseudo_h, pseudo_var):
        """
        Compute log E_{q(x)} [ N(y | H x, diag(var_diag)) ] for each n

        Inputs:
            y_pseudo:   (N, M)     - pseudo targets
            mean_x:     (N, D)     - mean of q(x_n)
            cov_x:      (N, D, D)  - covariance of q(x_n)
            H_pseudo:   (M, D)     - observation matrix
            var_diag:   (N, M)     - diagonal noise variance for each y_n

        Returns:
            log_probs:  (N,)       - per-step predictive log-likelihood
        """
        N, M = pseudo_y.shape
        D = mean_x.shape[1]

        # Projected mean: H @ mu_n → (N, M)
        mean_proj = mean_x @ pseudo_h.T

        # Projected covariance: H Σ_n Hᵗ + R_n → (N, M, M), diagonal only
        proj_cov_diag = torch.einsum("md,ndd,md->nm", pseudo_h, cov_x, pseudo_h)  # (N, M)
        total_var_diag = proj_cov_diag + pseudo_var  # (N, M)

        # log N(y | mean_proj, diag(total_var_diag))
        residual = pseudo_y - mean_proj  # (N, M)
        log_det = torch.log(total_var_diag).sum(-1)  # (N,)
        quad_form = (residual ** 2 / total_var_diag).sum(-1)  # (N,)

        log_probs = -0.5 * (log_det + quad_form + M * np.log(2 * np.pi))  # (N,)
        return log_probs.sum()

    ############# KL term (Done) #######################
    def cal_kl(self, mean_x, cov_x, pseudo_y, pseudo_var, pseudo_h, log_lik_pseudo):
        # input: mean_x, cov_x, pseudo_y, pseudo_var, log_lik_pseudo (term 3)
        # return: term2 - term 3

        expected_density_pseudo = self.expected_log_lik_diag(pseudo_y, mean_x, cov_x, pseudo_h, pseudo_var)

        kl = expected_density_pseudo - log_lik_pseudo

        return kl

    def neg_elbo(self, Y, t):
        Y = torch.tensor(Y, device=self.device, dtype=torch.double)
        # Step1: 计算dt
        if t is not None:
            dt = torch.tensor([0] + list(np.diff(t)))
        else:
            dt = 0

        # Step2: 计算 q(x) 的均值和 方差,
        # mean_x 的维度应该是 (N, D), cov_x 的维度则是, (N, D, D)
        # pseudo_h, (M, D) 我们直接学习 mean, 这个可以先去初始化
        # pseudo_y (N, M), 和 pseudo_var, (N, M)?, scalar:
        # log_lik_pseudo ? scalar, 是 term 3 的值

        mean_x, cov_x, pseudo_y, pseudo_var, pseudo_h, log_lik_pseudo = self.update_posterior(Y, dt)

        # mean_x = self.mu_x
        # cov_x = self.log_sigma_x  # shape: N * Q

        # Step3: 用 mean_x 和 cov_x 采样出来 x, 来计算 neg_log_likelihood, 这个要改变一下 compute loss 的计算, 先变一下
        neg_likelihood = self.neg_log_likelihood(Y, mean_x, cov_x)

        kl = self.cal_kl(mean_x, cov_x, pseudo_y, pseudo_var, pseudo_h, log_lik_pseudo)

        return neg_likelihood + (kl)

    def pred_x(self, Y, t):
        if t is not None:
            dt = torch.tensor([0] + list(np.diff(t)))
        else:
            dt = 0

        Y = torch.tensor(Y, device=self.device, dtype=torch.double)

        mean_x, cov_x, pseudo_y, pseudo_var, pseudo_h, log_lik_pseudo = self.update_posterior(Y, dt)

        return mean_x, cov_x

    def f_eval(self, batch_y, mean_x, cov_x, x_star=None):
        """
            evaluation of the latent mapping function

            x_star:         prediction input                            shape: N_star * Q
            batch_y:        observations for characterizing the GP      shape: N * obs_dim
        """
        batch_y = torch.tensor(batch_y, device=self.device, dtype=torch.double)

        if x_star is None:
            x_star = mean_x

        Phi, Phi_star = self._compute_sm_basis(mean_x, cov_x, x_star=x_star, f_eval=True)

        cross_matrix = Phi_star @ Phi.T                                  # shape: N_star * N

        k_matrix = self._compute_gram_approximate_2(Phi=Phi)             # shape: N * N
        C_matrix = k_matrix + self.log_noise.exp() * torch.eye(self.N, device=self.device)

        L = torch.cholesky(C_matrix)                                    # shape: N x N
        L_inv_y = triangular_solve(batch_y, L, upper=False)[0]          # inv(L) * y
        K_L_inv = triangular_solve(cross_matrix.T, L, upper=False)[0]   # inv(L) * K_{N, N_star}

        f_star = K_L_inv.T @ L_inv_y                          # shape: N_star * obs_dim

        return f_star, k_matrix
