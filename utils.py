import os
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import (CenterCrop, Compose,
                                    Normalize, Resize,
                                    ToTensor)


def compute_gradients(func, inp, **kwargs):
    """Compute gradients for a given input image

    Parameters
    ----------
    func : callable
        Function that takes in an input image as well as kwargs and returns a single element tensor
    inp : torch.tensor
        Tensor for which grad has to be computed
    **kwargs : dict
        Additional keyword arguments passed to func

    Returns
    -------
    grad : torch.tensor
        Same shape gradient wrt input image, representing each pixel location
    """

    inp.requires_grad = True

    loss = func(inp, **kwargs)
    loss.backward()

    inp.requires_grad = False
    return inp.grad.data


def read_image(path):
    """Loads image for a given path and converts it to torch.tensor

    Parameters
    ----------
    path : str
        Path of image to be read

    Returns
    -------
    tensor : torch.tensor
    """

    image = Image.open(path)
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.227, 0.224, 0.225])]
    )
    tensor_ = transform(image)
    tensor = tensor_.unsqueeze(0)
    return tensor


def to_array(tensor):
    """Converts a tensor to np array

    Parameters
    ----------
    tensor : torch.tensor
        Per image tensor (1, 3, *, *)

    Returns
    _______
    arr : np.ndarray
        Array of image (*, *, 3)
    """

    tensor_ = tensor.squeeze(0)
    unnormalize_transform = Compose([Normalize(mean=[0, 0, 0],
                                               std=[1/0.227, 1/0.224, 1/0.225]),
                                    Normalize(mean=[-0.485, -0.456, -0.406],
                                              std=[1, 1, 1])])
    tensor_ = unnormalize_transform(tensor_)
    arr = tensor_.permute(1, 2, 0).detach().numpy()
    return arr


def scale_gradients(grad, quantile=0.98):
    """Scales gardient tensor

    Parameters
    ----------
    grad : torch.tensor
        Gradient of shape (1, 3, *, *)
    quantile : float
        Quantile value for normalizing grad

    Returns
    -------
    grad_arr : np.ndarray
        Array of shape (*, *, 1)
    """
    grad_arr = torch.abs(grad).mean(dim=1).detach().permute(1, 2, 0)
    grad_arr /= grad_arr.quantile(quantile)
    grad_arr = torch.clamp(grad_arr, 0, 1)
    return grad_arr.numpy()


def visualize(np_arrays, titles, name="fig.png"):
    """Visualizes images

    Parameters
    ----------
    np_arrays : list : np.array
        list of np images
    titles : list
        titles per images
    name : str, optional
        name of plot to be saved, by default "fig.png"
    """
    n = len(np_arrays)
    _, axes = plt.subplots(1, n)

    for i in range(n):
        arr = np_arrays[i]
        axes[i].imshow(arr)
        axes[i].axis("off")
        axes[i].set_title(titles[i])

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{name}", bbox_inches="tight")


# if __name__ == "__main__":
#     x = torch.randn((1, 3, 10, 10))
#     gr_x = scale_gradients(x)
#     print(gr_x.shape)
