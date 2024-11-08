
import matplotlib.pyplot as plt
from src.data_utils import load_cifar10_data

def show_sample_images(x, y, num_samples=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x[i])
        plt.title(f"Label: {y[i][0]}")
        plt.axis('off')
    plt.show()

# Load CIFAR-10 data and display some samples
(x_train, y_train), _ = load_cifar10_data()
show_sample_images(x_train, y_train)
