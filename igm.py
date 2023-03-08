# IGM : Integrated gradient method
import torch
import torchvision.models as models
import numpy as np

from utils import compute_gradients, read_image, visualize, to_array, scale_gradients

import warnings
warnings.filterwarnings("ignore")


def func(inp, net=None, target=None):
    """Computes logits via forward pass

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
    logit : torch.Tensor
        logits of target class
    """
    out = net(inp)
    logit = out[0, target]
    return logit


def compute_ig(inp, baseline, net, target, n_steps=100):
    """Perform IGM method for corrupting image wrt a target class

    Parameters
    ----------
    inp : torch.Tensor
        input image (1, 3, *, *)
    baseline : torch.Tensor
        baseline image (1, 3, *, *)
    net : torch.nn.Module
        classifier network
    target : int
        ground truth label
    n_steps : int, optional
        Number of steps between inp and baseline, by default 100

    Returns
    -------
    ig : torch.Tensor
        integrated gradients (1, 3, *, *)
    inp_grad : torch.Tensor
        gradients wrt inp image
    """

    interpolation_path = [
        baseline+alpha*(inp-baseline) for alpha in np.linspace(0, 1, n_steps)]
    grads = [compute_gradients(func, x, net=net, target=target)
             for x in interpolation_path]

    ig = (inp-baseline)*torch.cat(grads[:-1]).mean(dim=0, keepdims=True)
    return ig, grads[-1]


if __name__ == "__main__":
    net = models.resnet18(pretrained=True)
    net.eval()

    path = "images/car.jpg"
    tensor = read_image(path)
    baseline = -1.5*torch.ones_like(tensor)
    target = 291

    ig, grads = compute_ig(inp=tensor, baseline=baseline,
                           net=net, target=target)

    ig_scaled = scale_gradients(ig)
    grads_scaled = scale_gradients(grads)

    org_arr = to_array(tensor)
    baseline_arr = to_array(baseline)
    grad_org_arr = org_arr * grads_scaled
    grad_ig = org_arr * ig_scaled

    visualize([org_arr, baseline_arr, grad_org_arr, grad_ig], titles=[
              "original", "baseline", "grad on original", "ig grads"],
              name=f"igm_visual_{path.split('/')[-1]}")
