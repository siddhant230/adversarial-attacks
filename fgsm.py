# FGSM : Fast Gradient Signed method
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models

from utils import compute_gradients, read_image, to_array, visualize

import warnings
warnings.filterwarnings("ignore")


def func(inp, net=None, target=None):
    """Computes NLL for given input

    Parameters
    ----------
    inp : torch.Tensor
        Input image; bs = 1
    net : torch.nn.module, optional
        Imagenet model, by default None
    target : int, optional
        Imagenet label id, by default None

    Returns
    -------
    loss : torch.Tensor
        loss wrt input image
    """

    out = net(inp)
    target = torch.LongTensor([target])
    loss = torch.nn.functional.nll_loss(out, target)
    print(f"loss: {loss.item()}")
    return loss


def attack(tensor, net, eps=1e-3, n_iter=50):
    """Iteratively performs FGSM attack on given input

    Parameters
    ----------
    tensor : torch.Tensor
        input image (1, 3, 224, 224)
    net : torch.nn.Module
        Trained neural net
    eps : float, optional
        per step modification criterion, by default 1e-3
    n_iter : int, optional
        number of iterations, by default 50

    Returns
    -------
    new_tensor : torch.Tensor
        modified/corrupted image after n iterations, that can fool model
    orig_pred : torch.Tensor
        actual prediction by model on uncorrupted image
    new_pred : torch.Tensor
        predictions by model on corrupted image
    """

    new_tensor = tensor.detach().clone()

    orig_pred = net(tensor).argmax()
    print(f"Original prediction : {orig_pred.item()}")

    for i in range(n_iter):
        net.zero_grad()

        grad = compute_gradients(func=func,
                                 inp=new_tensor,
                                 net=net,
                                 target=orig_pred.item())
        new_tensor = torch.clamp(new_tensor + eps*grad.sign(), -2, 2)
        new_pred = net(new_tensor).argmax()
        if orig_pred != new_pred:
            print(f"Fooled classifier after {i} iterations")
            print(f"New predictions : {new_pred.item()}")
            break

    return new_tensor, orig_pred.item(), new_pred.item()


if __name__ == "__main__":
    net = models.resnet18(pretrained=True)
    net.eval()
    path = "images/cat.jpeg"
    tensor = read_image(path)
    new_tensor, orig_pred, new_pred = attack(tensor, net)

    arr = to_array(tensor)
    new_arr = to_array(new_tensor)
    diff_arr = np.abs(arr - new_arr).mean(axis=2)   # across channel mean
    diff_arr = diff_arr/diff_arr.max()

    visualize([arr, new_arr, diff_arr], titles=["original image", "corrupted image", "difference"],
              name=f"fgsm_visual_{path.split('/')[-1]}")
