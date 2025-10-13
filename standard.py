"""
Implements the standard SAE training scheme with PCA whitening.
"""
import torch as t
from typing import Optional, Tuple
import numpy as np

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import AutoEncoder
from collections import namedtuple

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


class StandardTrainer(SAETrainer):
    """
    Standard SAE training scheme with PCA whitening preprocessing.
    The encoder operates on whitened activations, while the decoder reconstructs in the original space.
    """
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=AutoEncoder,
                 lr: float = 1e-3,
                 l1_penalty: float = 1e-1,
                 warmup_steps: int = 1000,
                 sparsity_warmup_steps: Optional[int] = 2000,
                 decay_start: Optional[int] = None,
                 resample_steps: Optional[int] = None,
                 seed: Optional[int] = None,
                 device=None,
                 wandb_name: Optional[str] = 'StandardTrainer',
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
        
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device

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

        # Initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)
        self.ae.to(self.device)

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def fit_whitener(self, activations: t.Tensor):
        """
        Fit the PCA whitener on a batch of activations.
        Should be called before training starts.
        """
        if self.use_pca_whitening and not self.whitener.is_fitted:
            print("Fitting PCA whitener...")
            self.whitener.fit(activations.to(self.device))
            print("PCA whitener fitted successfully")

    def forward_with_whitening(self, x: t.Tensor, output_features: bool = False) -> Tuple[t.Tensor, Optional[t.Tensor]]:
        """
        Forward pass with PCA whitening.
        
        Args:
            x: Input activations in original space
            output_features: Whether to return latent features
        Returns:
            x_hat: Reconstruction in original space
            f: Latent features (if output_features=True)
        """
        # Check if we should and can use whitening
        if self.use_pca_whitening and self.whitener is not None and self.whitener.is_fitted:
            # Whiten input
            x_whitened = self.whitener.whiten(x)
            
            # Pass through encoder (in whitened space)
            f = self.ae.encode(x_whitened)
            
            # Decode to whitened space
            x_hat_whitened = self.ae.decode(f)
            
            # Dewhiten reconstruction to original space
            x_hat = self.whitener.dewhiten(x_hat_whitened)
        else:
            # Standard forward pass without whitening (or if whitener not fitted yet)
            x_hat, f = self.ae(x, output_features=True)
        
        if output_features:
            return x_hat, f
        return x_hat

    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: 
                return
            print(f"resampling {deads.sum().item()} neurons")

            # Compute loss for each activation
            x_hat, _ = self.forward_with_whitening(activations, output_features=True)
            losses = (activations - x_hat).norm(dim=-1)

            # Sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]
            
            # If using whitening, transform sampled vectors
            if self.use_pca_whitening and self.whitener.is_fitted:
                sampled_vecs_whitened = self.whitener.whiten(sampled_vecs)
            else:
                sampled_vecs_whitened = sampled_vecs

            # Get norm of the living neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # Resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs_whitened * alive_norm * 0.2
            
            # For decoder, we need to account for whitening transformation
            if self.use_pca_whitening and self.whitener.is_fitted:
                # Decoder operates in whitened space
                sampled_vecs_whitened_norm = sampled_vecs_whitened / sampled_vecs_whitened.norm(dim=-1, keepdim=True)
                self.ae.decoder.weight[:, deads] = sampled_vecs_whitened_norm.T
            else:
                self.ae.decoder.weight[:, deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            
            self.ae.encoder.bias[deads] = 0.

            # Reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:, deads] = 0.
            state_dict[3]['exp_avg_sq'][:, deads] = 0.
    
    def loss(self, x, step: int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)

        # Forward pass with whitening
        x_hat, f = self.forward_with_whitening(x, output_features=True)
        
        # Compute losses in original space
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        if self.steps_since_active is not None:
            # Update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'sparsity_loss': l1_loss.item(),
                    'loss': loss.item()
                }
            )

    def update(self, step, activations):
        activations = activations.to(self.device)
        
        # Collect batches for PCA fitting if needed
        if self.use_pca_whitening and not self.whitener.is_fitted:
            self.pca_collection_buffer.append(activations.clone())
            
            # Check if we've collected enough batches
            if len(self.pca_collection_buffer) >= self.pca_n_batches:
                # Concatenate all collected batches
                all_activations = t.cat(self.pca_collection_buffer, dim=0)
                print(f"Collected {len(self.pca_collection_buffer)} batches ({all_activations.shape[0]} samples total)")
                
                # Fit the whitener
                self.fit_whitener(all_activations)
                
                # Clear the buffer to free memory
                self.pca_collection_buffer = None
                
                # Process the current batch normally
                self.optimizer.zero_grad()
                loss = self.loss(activations, step=step)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
                # Still collecting batches, skip training update
                print(f"Collecting PCA statistics: {len(self.pca_collection_buffer)}/{self.pca_n_batches} batches")
                return
        else:
            # Normal training update (after PCA is fitted or when not using PCA)
            self.optimizer.zero_grad()
            loss = self.loss(activations, step=step)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class': 'StandardTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'l1_penalty': self.l1_penalty,
            'warmup_steps': self.warmup_steps,
            'resample_steps': self.resample_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'use_pca_whitening': self.use_pca_whitening,
        }


class StandardTrainerAprilUpdate(SAETrainer):
    """
    Standard SAE training scheme following the Anthropic April update with PCA whitening.
    Decoder column norms are NOT constrained to 1.
    """
    def __init__(self,
                 steps: int,
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=AutoEncoder,
                 lr: float = 1e-3,
                 l1_penalty: float = 1e-1,
                 warmup_steps: int = 1000,
                 sparsity_warmup_steps: Optional[int] = 2000,
                 decay_start: Optional[int] = None,
                 seed: Optional[int] = None,
                 device=None,
                 wandb_name: Optional[str] = 'StandardTrainerAprilUpdate',
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

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device

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

        # Initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)
        self.ae.to(self.device)

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None, sparsity_warmup_steps=sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def fit_whitener(self, activations: t.Tensor):
        """
        Fit the PCA whitener on a batch of activations.
        """
        if self.use_pca_whitening and not self.whitener.is_fitted:
            print("Fitting PCA whitener...")
            self.whitener.fit(activations.to(self.device))
            print("PCA whitener fitted successfully")

    def forward_with_whitening(self, x: t.Tensor, output_features: bool = False) -> Tuple[t.Tensor, Optional[t.Tensor]]:
        """
        Forward pass with PCA whitening.
        """
        # Check if we should and can use whitening
        if self.use_pca_whitening and self.whitener is not None and self.whitener.is_fitted:
            # Whiten input
            x_whitened = self.whitener.whiten(x)
            
            # Pass through encoder (in whitened space)
            f = self.ae.encode(x_whitened)
            
            # Decode to whitened space
            x_hat_whitened = self.ae.decode(f)
            
            # Dewhiten reconstruction to original space
            x_hat = self.whitener.dewhiten(x_hat_whitened)
        else:
            # Standard forward pass without whitening (or if whitener not fitted yet)
            x_hat, f = self.ae(x, output_features=True)
        
        if output_features:
            return x_hat, f
        return x_hat

    def loss(self, x, step: int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)

        # Forward pass with whitening
        x_hat, f = self.forward_with_whitening(x, output_features=True)
        
        # Compute losses in original space
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # L1 loss with decoder norm (April update style)
        # Note: decoder weights are in whitened space if using PCA
        l1_loss = (f * self.ae.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()

        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'sparsity_loss': l1_loss.item(),
                    'loss': loss.item()
                }
            )

    def update(self, step, activations):
        activations = activations.to(self.device)
        
        # Collect batches for PCA fitting if needed
        if self.use_pca_whitening and not self.whitener.is_fitted:
            self.pca_collection_buffer.append(activations.clone())
            
            # Check if we've collected enough batches
            if len(self.pca_collection_buffer) >= self.pca_n_batches:
                # Concatenate all collected batches
                all_activations = t.cat(self.pca_collection_buffer, dim=0)
                print(f"Collected {len(self.pca_collection_buffer)} batches ({all_activations.shape[0]} samples total)")
                
                # Fit the whitener
                self.fit_whitener(all_activations)
                
                # Clear the buffer to free memory
                self.pca_collection_buffer = None
                
                # Process the current batch normally
                self.optimizer.zero_grad()
                loss = self.loss(activations, step=step)
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            else:
                # Still collecting batches, skip training update
                print(f"Collecting PCA statistics: {len(self.pca_collection_buffer)}/{self.pca_n_batches} batches")
                return
        else:
            # Normal training update (after PCA is fitted or when not using PCA)
            self.optimizer.zero_grad()
            loss = self.loss(activations, step=step)
            loss.backward()
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class': 'StandardTrainerAprilUpdate',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'l1_penalty': self.l1_penalty,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'use_pca_whitening': self.use_pca_whitening,
        }
