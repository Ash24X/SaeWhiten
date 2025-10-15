"""
Implements the SAE training scheme from https://arxiv.org/abs/2406.04093 with PCA whitening.
Significant portions of this code have been copied from https://github.com/EleutherAI/sae/blob/main/sae
"""

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from typing import Optional, Tuple

from ..config import DEBUG
from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


class PCAWhitener:
    """
    PCA whitening transformation for input activations.
    """
    def __init__(self, activation_dim: int, n_samples: int = 10000, eps: float = 1e-6, device='cuda'):
        self.activation_dim = activation_dim
        self.n_samples = n_samples
        self.eps = eps
        self.device = device
        
        # Whitening parameters (will be computed from data)
        self.mean = None
        self.whitening_matrix = None
        self.dewhitening_matrix = None
        self.is_fitted = False
    
    def fit(self, activations: t.Tensor):
        """
        Compute PCA whitening parameters from a batch of activations.
        
        Args:
            activations: Tensor of shape (batch_size, activation_dim)
        """
        with t.no_grad():
            # Use a subset of activations if batch is too large
            if activations.shape[0] > self.n_samples:
                indices = t.randperm(activations.shape[0])[:self.n_samples]
                activations = activations[indices]
            
            # Compute mean
            self.mean = activations.mean(dim=0)
            
            # Center the data
            centered = activations - self.mean
            
            # Compute covariance matrix
            cov = (centered.T @ centered) / (activations.shape[0] - 1)
            
            # Add small epsilon to diagonal for numerical stability
            cov = cov + self.eps * t.eye(self.activation_dim, device=self.device)
            
            # Compute eigendecomposition
            eigenvalues, eigenvectors = t.linalg.eigh(cov)
            
            # Sort in descending order
            idx = eigenvalues.argsort(descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Compute whitening matrix: W = D^(-1/2) @ E^T
            # where D is diagonal matrix of eigenvalues, E is eigenvector matrix
            D_sqrt_inv = t.diag(1.0 / t.sqrt(eigenvalues + self.eps))
            self.whitening_matrix = D_sqrt_inv @ eigenvectors.T
            
            # Compute dewhitening matrix (inverse transformation)
            D_sqrt = t.diag(t.sqrt(eigenvalues + self.eps))
            self.dewhitening_matrix = eigenvectors @ D_sqrt
            
            self.is_fitted = True
            
            # Print statistics
            print(f"PCA Whitening fitted on {activations.shape[0]} samples")
            print(f"Eigenvalue range: [{eigenvalues.min().item():.6f}, {eigenvalues.max().item():.6f}]")
            print(f"Condition number: {(eigenvalues.max() / eigenvalues.min()).item():.2f}")
    
    def whiten(self, x: t.Tensor) -> t.Tensor:
        """
        Apply whitening transformation.
        
        Args:
            x: Input tensor of shape (batch_size, activation_dim)
        Returns:
            Whitened tensor of same shape
        """
        if not self.is_fitted:
            raise RuntimeError("PCAWhitener must be fitted before whitening")
        
        # Center and whiten
        x_centered = x - self.mean
        x_whitened = x_centered @ self.whitening_matrix.T
        return x_whitened
    
    def dewhiten(self, x_whitened: t.Tensor) -> t.Tensor:
        """
        Apply inverse whitening transformation.
        
        Args:
            x_whitened: Whitened tensor of shape (batch_size, activation_dim)
        Returns:
            Original space tensor of same shape
        """
        if not self.is_fitted:
            raise RuntimeError("PCAWhitener must be fitted before dewhitening")
        
        # Dewhiten and uncenter
        x_centered = x_whitened @ self.dewhitening_matrix.T
        x = x_centered + self.mean
        return x


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class AutoEncoderTopK(Dictionary, nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the AutoEncoderTopK class docstring in that branch.

    With some additional effort, you can train a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(
        self, x: t.Tensor, return_topk: bool = False, use_threshold: bool = False
    ):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            if return_topk:
                post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
                return (
                    encoded_acts_BF,
                    post_topk.values,
                    post_topk.indices,
                    post_relu_feat_acts_BF,
                )
            else:
                return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    def from_pretrained(path, k: Optional[int] = None, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class TopKTrainer(SAETrainer):
    """
    Top-K SAE training scheme with PCA whitening preprocessing.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = AutoEncoderTopK,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,  # see Appendix A.2
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        k_anneal_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "AutoEncoderTopK",
        submodule_name: Optional[str] = None,
        use_pca_whitening: bool = True,
        pca_n_batches: int = 10,  # Number of batches to collect for PCA
        pca_eps: float = 1e-6,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.k_anneal_steps = k_anneal_steps

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialise autoencoder
        self.ae = dict_class(activation_dim, dict_size, k)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        # PCA Whitening setup
        self.use_pca_whitening = use_pca_whitening
        self.pca_n_batches = pca_n_batches
        if self.use_pca_whitening:
            self.whitener = PCAWhitener(
                activation_dim=activation_dim,
                n_samples=50000,  # Will collect ~20k samples from multiple batches
                eps=pca_eps,
                device=self.device
            )
            self.pca_collection_buffer = []  # Buffer to collect batches for PCA fitting
        else:
            self.whitener = None
            self.pca_collection_buffer = None

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def fit_whitener(self, activations: t.Tensor):
        """
        Fit the PCA whitener on a batch of activations.
        Should be called before training starts.
        """
        if self.use_pca_whitening and not self.whitener.is_fitted:
            print("Fitting PCA whitener...")
            self.whitener.fit(activations.to(self.device))
            print("PCA whitener fitted successfully")

    def forward_with_whitening(self, x: t.Tensor, return_topk: bool = False, use_threshold: bool = False) -> Tuple:
        """
        Forward pass with PCA whitening.
        
        Args:
            x: Input activations in original space
            return_topk: Whether to return top-k information
            use_threshold: Whether to use thresholding
        Returns:
            Depending on return_topk:
            - If False: (x_hat,) reconstruction in original space
            - If True: (x_hat, f, top_acts_BK, top_indices_BK, post_relu_acts_BF)
        """
        # Check if we should and can use whitening
        if self.use_pca_whitening and self.whitener is not None and self.whitener.is_fitted:
            # Whiten input
            x_whitened = self.whitener.whiten(x)
            
            # Pass through encoder (in whitened space)
            if return_topk:
                f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
                    x_whitened, return_topk=True, use_threshold=use_threshold
                )
            else:
                f = self.ae.encode(x_whitened, return_topk=False, use_threshold=use_threshold)
            
            # Decode to whitened space
            x_hat_whitened = self.ae.decode(f)
            
            # Dewhiten reconstruction to original space
            x_hat = self.whitener.dewhiten(x_hat_whitened)
        else:
            # Standard forward pass without whitening (or if whitener not fitted yet)
            if return_topk:
                f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
                    x, return_topk=True, use_threshold=use_threshold
                )
            else:
                f = self.ae.encode(x, return_topk=False, use_threshold=use_threshold)
            
            x_hat = self.ae.decode(f)
        
        if return_topk:
            return x_hat, f, top_acts_BK, top_indices_BK, post_relu_acts_BF
        else:
            return x_hat

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None:
            return

        assert 0 <= k_anneal_steps < self.steps, (
            "k_anneal_steps must be >= 0 and < steps."
        )
        # self.k is the target k set for the trainer, not the dictionary's current k
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        # Update in-place
        self.ae.k.fill_(int(annealed_value))

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: For auxiliary loss, we need to handle whitening carefully
            if self.use_pca_whitening and self.whitener is not None and self.whitener.is_fitted:
                # Decode in whitened space, then dewhiten
                x_reconstruct_aux_whitened = self.ae.decoder(auxk_acts_BF)
                x_reconstruct_aux = self.whitener.dewhiten(x_reconstruct_aux_whitened)
            else:
                # Standard decode without bias (decoder() not decode())
                x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, top_acts_BK: t.Tensor):
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=t.float32)
            min_activation = min_activations.mean()

            B, K = active.shape
            assert len(active.shape) == 2
            assert min_activations.shape == (B,)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        # Run the SAE with whitening
        x_hat, f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.forward_with_whitening(
            x, return_topk=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        # Measure goodness of reconstruction (in original space)
        e = x - x_hat

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
            if self.auxk_alpha > 0
            else 0
        )

        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        x = x.to(self.device)
        
        # Collect batches for PCA fitting if needed
        if self.use_pca_whitening and not self.whitener.is_fitted:
            self.pca_collection_buffer.append(x.clone())
            
            # Check if we've collected enough batches
            if len(self.pca_collection_buffer) >= self.pca_n_batches:
                # Concatenate all collected batches
                all_activations = t.cat(self.pca_collection_buffer, dim=0)
                print(f"Collected {len(self.pca_collection_buffer)} batches ({all_activations.shape[0]} samples total)")
                
                # Fit the whitener
                self.fit_whitener(all_activations)
                
                # Clear the buffer to free memory
                self.pca_collection_buffer = None
                
                # Initialize decoder bias with geometric median (in original space)
                median = geometric_median(all_activations)
                median = median.to(self.ae.b_dec.dtype)
                self.ae.b_dec.data = median
                
                # Process the current batch normally
                loss = self.loss(x, step=step)
                loss.backward()
                
                # clip grad norm and remove grads parallel to decoder directions
                self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                    self.ae.decoder.weight,
                    self.ae.decoder.weight.grad,
                    self.ae.activation_dim,
                    self.ae.dict_size,
                )
                t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                
                # do a training step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)
                
                # Make sure the decoder is still unit-norm
                self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                    self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
                )
                
                return loss.item()
            else:
                # Still collecting batches, skip training update
                print(f"Collecting PCA statistics: {len(self.pca_collection_buffer)}/{self.pca_n_batches} batches")
                return 0.0  # Return dummy loss value
        
        # Normal training (after PCA is fitted or when not using PCA)
        
        # Initialise the decoder bias on first step (only if not using PCA, otherwise it's done after PCA fitting)
        if step == 0 and not self.use_pca_whitening:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # compute the loss
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "TopKTrainer",
            "dict_class": "AutoEncoderTopK",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "use_pca_whitening": self.use_pca_whitening,
        }
