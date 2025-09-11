import numpy as np
import scipy.io as sio
from WLSDL import WLSDL
from result_show import result_show
import HyperProTool as hpt

# Load example dataset (AVIRIS San Diego)
data = sio.loadmat(r'C:\Users\hp\Hyperspectral-Anomaly-Detection\SanDiego.mat')  # Update path
Y = data['Sandiego']  # Hyperspectral data: 400x400x224
# Reshape from (400, 400, 224) to (224, 160000)
height, width, bands = Y.shape
Y = Y.reshape(height * width, bands).T  # To bands x pixels (224 x 160000)

# Optional: Crop to 100x100 pixels for faster processing
crop = True  # Set to False to use full 400x400
if crop:
    Y = Y[:, :10000]  # First 100x100 pixels (10000 columns)
    height, width = 100, 100  # Update for visualization

groundTruth = None  # No ground truth available

# Parameters for WLSDL (small for your CPU)
lambda1 = 0.01
alpha = 0.1
beta = 0.01
theta = 0.5

k = 50  # Small for testing
max_iter = 100

# Run WLSDL
D, A, E = WLSDL(Y, lambda1=lambda1, alpha=alpha, beta=beta, theta=theta, k=k, max_iter=max_iter)

# Compute anomaly map from residual
anomaly_map = np.linalg.norm(E, axis=0)

# Visualize results (no ground truth)
result_show(Y, D, A, E, groundTruth, height=height, width=width)

print("No ground truth available; showing anomaly map only.")