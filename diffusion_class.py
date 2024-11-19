import torch
import torch.nn as nn

class Diffusion: 
    def __init__(self, T=1000, beta_min=10e-5, beta_max=0.02, schedule='linear'):
        """
        Initialize the diffusion process.
        Args:
            T: Total number of timesteps.
            beta_min: Minimum value of beta in the noise schedule.
            beta_max: Maximum value of beta in the noise schedule.
            schedule: Type of noise schedule ('linear', 'cosine', etc.).
        """
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.schedule = schedule

        # Noise schedule parameters
        self.beta = None
        self.alpha = None
        self.alpha_bar = None

        # Precompute schedule
        self.get_noise_schedule()

    def get_noise_schedule(self):
        """
        Precompute the noise schedule.
        """
        if self.schedule == 'linear':
            self.beta = torch.linspace(self.beta_min, self.beta_max, self.T)
            self.alpha = 1 - self.beta
            self.alpha_bar = self.alpha.cumprod(dim=0)

        elif self.schedule == 'cosine':
            raise NotImplementedError("Cosine schedule is not implemented yet.")
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule}")
        
    
    def forward_diffusion(self, x0, t):
        """
        Perform the forward diffusion process.
        Args:
            x0: Original data (e.g., images).
            t: Timesteps (tensor of integers).
        Returns:
            xt: Noisy data at timestep t.
            epsilon: The noise added.
        """
        eps = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)

        print(sqrt_alpha_bar_t.shape)
        print(sqrt_one_minus_alpha_bar_t.shape)
        print(x0.shape)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps
 
        return xt, eps 
    
    def reverse_diffusion(self, xt, t, model):
        pass 



    