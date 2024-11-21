"""
model_unet.py

This module defines the architecture and components of a U-Net model using PyTorch. 
The model will be used as the neural network used to predict the noise in DDPMs. 

WORK IN PROGRESS
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 


### Positional Encoding ###
class SinusoidalPositionEmbeddings(nn.Module):
    '''
    From spmallick on GitHub

    - chose to use this over own implementation as it is more efficient (due to precomputing the embeddings, allowing for faster training)
    '''
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)


### SUBMODULES ### 
# Block for self attention mechanism
class SelfAttentionBlock(nn.Module):
    # TODO - implement self attention mechanism
    
    pass 


# Block for downsampling part of U-net
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dims):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.gnorm = nn.GroupNorm(8, out_channels//2)
        self.activation = nn.SiLU()
        self.downsample = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=2, padding=1)
        self.time_proj = nn.Linear(time_emb_dims, out_channels//2)

    def forward(self, x, time_emb):
        # Add timestep embedding
        time_emb_proj = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        # print(f"Before adding time: x shape: {x.shape}, time_emb shape: {time_emb_proj.shape}")
        # print(f"The input to the downsample block is: {x.shape}")
        x = self.conv(x) + time_emb_proj
        x = self.gnorm(x)
        x = self.activation(x)
        skip = x  # Save for skip connection
        #print(f"Right before DS: x shape: {x.shape}, skip shape: {skip.shape}")
        x = self.downsample(x)
        #print(f"After DS: x shape: {x.shape}, skip shape: {skip.shape}")
        return x, skip

# Block for bottleneck part of U-net
class BottleneckBlock(nn.Module):
    def __init__(self, channels, time_emb_dims):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.activation = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dims, channels)

    def forward(self, x, time_emb):
        #print(f"Before bottleneck: x shape: {x.shape}")
        time_emb_proj = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.conv(x) + time_emb_proj
        x = self.norm(x)
        #print(f"After bottleneck: x shape: {x.shape}")
        return self.activation(x) 


# Block for upsampling part of U-net
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dims):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        self.upsample = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=4, stride=2, padding=1)
        self.time_proj = nn.Linear(time_emb_dims, out_channels)

    def forward(self, x, skip, time_emb):
        # Upsample
        #print(f"Before US: x shape: {x.shape}, skip shape: {skip.shape}")

        x = self.upsample(x)
        #print(f"executed")
        #print(f"After US: x shape: {x.shape}, skip shape: {skip.shape}")

        # Resize if spatial dimensions mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            #print("Activated interpolation")
        
        #print(f"After interpolation: x shape: {x.shape}, skip shape: {skip.shape}")
        # Add skip connection
        x = torch.cat([x, skip], dim=1)
        
        # 3. Reduce concatenated channels
        #print(f"Before conv: x shape: {x.shape}")
        x = self.conv(x)

        # Add timestep embedding
        time_emb_proj = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + time_emb_proj
        x = self.norm(x)
        x = self.activation(x)
        return x



### U-NET MODEL ###

class UNet(nn.Module):
    def __init__(self, input_channels=1, resolutions=[64, 128, 256, 512], time_emb_dims=512):
        """
        U-Net implementation for DDPM
        Args:
            input_channels: Number of input channels (e.g., 1 for MNIST).
            base_channels: Number of channels in the first convolution layer.
            num_resolutions: Number of downsampling/upsampling blocks.
            time_emb_dims: Dimensionality of timestep embeddings.
        """
        super().__init__()
        
        self.base_channels = resolutions[0]


        # Input layer
        self.input_conv = nn.Conv2d(input_channels, self.base_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels=resolutions[i],
                out_channels=resolutions[i+1],
                time_emb_dims=time_emb_dims
            )
            for i in range(len(resolutions) - 1)
        ])


        # Bottleneck block
        self.bottleneck = BottleneckBlock(channels=resolutions[-1], time_emb_dims=time_emb_dims)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels=resolutions[i + 1],
                out_channels=resolutions[i],
                time_emb_dims=time_emb_dims
            )
            for i in reversed(range(len(resolutions) - 1))
        ])

        # Output layer 
        self.output_conv = nn.Conv2d(self.base_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x, time_emb):
        """
        Forward pass of the U-Net.
        Args:
            x: Input image tensor of shape (batch_size, input_channels, height, width).
            time_emb: Timestep embedding tensor of shape (batch_size, time_emb_dims).
        Returns:
            Tensor of shape (batch_size, input_channels, height, width).
        """

        # Input conv 
        x = self.input_conv(x)

        # Downsampling path 
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, time_emb)
            skips.append(skip)
        
        #print(f"Skips: {len(skips)}")
        # Bottleneck block
        x = self.bottleneck(x, time_emb)

        # Upsampling path
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip, time_emb) 
        
        # Output conv
        x = self.output_conv(x)
        
        return x


if __name__ == "__main__":
    # Test the framework

    # Step 1: Initialize the Sinusoidal Embeddings
    time_embedding = SinusoidalPositionEmbeddings(total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512)
    
    # Step 2: Initialize the U-Net
    unet = UNet(input_channels=1, resolutions=[64, 128, 256, 512], time_emb_dims=512)
    
    # Step 3: Create dummy inputs
    batch_size = 8
    height, width = 28, 28  # For MNIST
    x = torch.randn(batch_size, 1, height, width)  # Random input image
    time = torch.randint(0, 1000, (batch_size,))  # Random timesteps for the batch

    # Step 4: Generate timestep embeddings
    time_emb = time_embedding(time)
    
    # Step 5: Forward pass through the U-Net
    output = unet(x, time_emb)

    # Step 6: Verify output shape
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == x.shape, "Output shape does not match input shape!"
    print("U-Net test passed successfully!")