import torch
import torch.nn as nn

class Diffusion: 
    def __init__(self, T=1000, beta_min=10e-5, beta_max=0.02, schedule='linear', device='cpu'):
        """
        Initialize the diffusion process.
        Args:
            T: Total number of timesteps.
            beta_min: Minimum value of beta in the noise schedule.
            beta_max: Maximum value of beta in the noise schedule.
            schedule: Type of noise schedule ('linear', 'cosine', etc.).
            device: Device to use for computations.
        """
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.schedule = schedule
        self.device = device

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
            self.beta = torch.linspace(self.beta_min, self.beta_max, self.T).to(self.device)
            self.alpha = (1 - self.beta).to(self.device)
            self.alpha_bar = self.alpha.cumprod(dim=0).to(self.device)

        elif self.schedule == 'cosine':
            steps = torch.linspace(0, torch.pi, self.T).to(self.device)
            self.beta = ((torch.cos(steps) + 1) * 0.5 * (self.beta_max - self.beta_min) + self.beta_min).flip(0)
            self.alpha = (1 - self.beta).to(self.device)
            self.alpha_bar = self.alpha.cumprod(dim=0).to(self.device)

            # s = 0.008  # offset
            # t = torch.linspace(0, self.T, self.T).to(self.device)
            # self.alpha_bar = torch.cos(((t / self.T) + s) / (1 + s) * np.pi/2) ** 2 # Cosine schedule
            # self.alpha_bar = self.alpha_bar/self.alpha_bar[0]
            # self.beta = 1 - self.alpha_bar/(self.alpha_bar -1)
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
        
        # print(f"alpha shape{self.alpha_bar[t].shape}")
        # print(sqrt_alpha_bar_t.shape)
        # print(sqrt_one_minus_alpha_bar_t.shape)
        # print(x0.shape)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps
 
        return xt, eps 
    
    def reverse_diffusion(self, xt, t, model, time_embedding):
        """
        Perform the reverse diffusion process.
        Args:
            xt: Noisy data at timestep t.
            t: Timesteps for batch (tensor of integers).
            model: Model used for reverse diffusion.
            time_embedding: Time embedding object.
        Returns:
            xr: Reconstructed data at timestep t.
        """
        # z = torch.randn_like(xt) if t > 1 else torch.zeros_like(xt)
        z = torch.where((t > 1).view(-1, 1, 1, 1), torch.randn_like(xt), torch.zeros_like(xt)) 
        
        # extract time embedding
        time_emb = time_embedding(t)

        # predict noise
        eps_theta = model(xt, time_emb)

        sqrt_alpha_t = self.alpha[t].sqrt().view(-1, 1, 1, 1)  # we need to reshape the tensor to match the shape of xt (same goes for the other tensors)
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1) 
        sigma_t = beta_t.sqrt() 

        xt_minus_one = 1. / sqrt_alpha_t * (xt - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta) + sigma_t * z

        return xt_minus_one 

        
  



    