import torch
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# — Pure PyTorch grid
step = 0.01
x_vals = torch.arange(-4.0, 4.0, step, device=device, dtype=torch.float32)
y_vals = torch.arange(-4.0, 4.0, step, device=device, dtype=torch.float32)
# meshgrid with indexing='ij' so dim-0 corresponds to y, dim-1 to x (matches np.mgrid layout)
y, x = torch.meshgrid(y_vals, x_vals, indexing='ij')


# Compute Gaussian
z = torch.exp(-(x**2 + y**2) / 2.0)

#plot
import matplotlib.pyplot as plt
plt.imshow(z.cpu().numpy())#Updated!
plt.tight_layout()
plt.show()
