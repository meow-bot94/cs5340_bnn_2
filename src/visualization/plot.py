import numpy as np
import matplotlib.pyplot as plt


def imshow_tensor(tensor):
    """Imshow for Tensor."""

    # Set the color channel as the third dimension
    image = tensor.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.axis('off')
