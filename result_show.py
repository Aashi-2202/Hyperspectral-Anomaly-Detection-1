import numpy as np
import matplotlib.pyplot as plt

def result_show(Y, D, A, E, groundTruth, height=100, width=100):
    background2d = np.dot(D, A).reshape(height, width, Y.shape[0]).transpose(2, 0, 1)
    anomaly_map = np.linalg.norm(E, axis=0).reshape(height, width)

    # False color image (adjust bands for RGB)
    false_color = np.stack([Y[50], Y[30], Y[10]], axis=-1)
    false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(false_color)
    plt.title('False Color Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(anomaly_map, cmap='jet')
    plt.title('Anomaly Map (Residual)')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)  # Ensure plot displays on Windows