"""
Deep Leakage from Gradients DLG attack.

MIT License

Copyright (c) 2019 MIT HAN Lab

@inproceedings{zhu19deep,
  title={Deep Leakage from Gradients},
  author={Zhu, Ligeng and Liu, Zhijian and Han, Song},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
"""
import random
import os
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------- #
# Data Loading.
# ---------------------------------------------------- #

cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]


def load_cifar100(batch_size, augmentations=True, data_path='~/data', shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)
    trainset, validset = _build_cifar100(
        path, augmentations, normalize)
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                        reduce=None, reduction='mean')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(batch_size, len(trainset)),
                                              shuffle=False, drop_last=False)

    return loss_fn, trainloader, validloader


def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _get_meanstd(dataset):
    cc = torch.cat([dataset[i][0].reshape(3, -1)
                    for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std

# ---------------------------------------------------- #
# Model architecture.
# ---------------------------------------------------- #


def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


def construct_model(num_classes=10, num_channels=3, seed=None):
    """Return model and initialization seed"""
    model_init_seed = seed if seed is not None else np.random.randint(
        0, 2**32 - 10)
    model = LeNetZhu(num_channels=num_channels, num_classes=num_classes)
    print(f'Model initialized with random key {model_init_seed}.')
    return model, model_init_seed


class LeNetZhu(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, kernel_size=5,
                      padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ---------------------------------------------------- #
# Data visualization.
# ---------------------------------------------------- #

def display_batch_iterative(data, fig=None, axes=None, denormalize=False, mean=None, std=None):
    """Updates images in a single window dynamically without opening multiple tabs.

    Args:
        data: Tensor of shape (batch_size, 3, H, W).
        fig: (optional) Matplotlib figure to reuse.
        axes: (optional) Matplotlib axes to reuse.
        denormalize (bool, optional): If True, denormalize the image tensor. Defaults to False.
        mean (tuple, optional): Mean for denormalization.  Required if denormalize is True.
        std (tuple, optional): Std for denormalization. Required if denormalize is True.

    Returns:
        fig, axes: The figure and axes for reuse.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a torch.Tensor")
    if data.ndim != 4:
        raise ValueError(
            "Input data must be a 4D tensor (batch_size, 3, H, W)")

    batch_size = data.shape[0]
    max_per_row = 10
    ncols = min(max_per_row, batch_size)
    nrows = math.ceil(batch_size / max_per_row)

    if fig is None or axes is None:
        plt.ion()  # Enable interactive mode
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))

        if nrows == 1:
            axes = [axes]  # make it a list with one row
        if ncols == 1:
            axes = [[ax] for ax in axes]

    try:
        axes = axes.tolist()
    except:
        pass

    for idx in range(batch_size):
        row = idx // max_per_row
        col = idx % max_per_row
        ax = axes[row][col]

        # Denormalize if required
        if denormalize:
            if mean is None or std is None:
                raise ValueError(
                    "Mean and std must be provided for denormalization.")
            img = data[idx].cpu() * torch.tensor(std).view(3, 1, 1) + \
                torch.tensor(mean).view(3, 1, 1)
            # Ensure values are in the [0, 1] range
            img = torch.clamp(img, 0, 1)
        else:
            img = data[idx].cpu()

        pil_img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
        ax.imshow(pil_img)
        ax.axis('off')

    # Hide any unused axes if total_images doesn't fill the grid completely
    for idx in range(batch_size, nrows * ncols):
        row = idx // max_per_row
        col = idx % max_per_row
        axes[row][col].axis('off')

    fig.canvas.draw()  # Redraw the figure
    fig.canvas.flush_events()  # Process UI events

    return fig, axes  # Return the figure and axes for reuse


def display_batch(data, denormalize=False, mean=None, std=None, block=True):
    """Displays the target images in a grid.

    Args:
        data: Tensor of shape (batch_size, 3, H, W).
        denormalize (bool, optional): If True, denormalize the image tensor. Defaults to False.
        mean (tuple, optional): Mean for denormalization.  Required if denormalize is True.
        std (tuple, optional): Std for denormalization. Required if denormalize is True.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a torch.Tensor")
    if data.ndim != 4:
        raise ValueError(
            "Input data must be a 4D tensor (batch_size, 3, H, W)")

    batch_size = data.shape[0]
    max_per_row = 10
    ncols = min(max_per_row, batch_size)
    nrows = math.ceil(batch_size / max_per_row)
    plt.figure(figsize=(ncols * 1.5, nrows * 1.5))

    for idx in range(batch_size):
        plt.subplot(nrows, ncols, idx + 1)

        # Denormalize if required
        if denormalize:
            if mean is None or std is None:
                raise ValueError(
                    "Mean and std must be provided for denormalization.")
            img = data[idx].cpu() * torch.tensor(std).view(3, 1, 1) + \
                torch.tensor(mean).view(3, 1, 1)
            # Ensure values are in the [0,1] range
            img = torch.clamp(img, 0, 1)
        else:
            img = data[idx].cpu()

        pil_img = transforms.ToPILImage()(img)
        plt.imshow(pil_img)
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=block)  # Show the figure

# ---------------------------------------------------- #
# DLG attack.
# ---------------------------------------------------- #


class DLG:
    def __init__(self, model, loss_fn, mean_std=(0.0, 1.0), batch_size=1, epochs=100, idlg=True,):
        """
        A simple DLG optimizer with a fixed step size.
        """
        self.model = model

        self.setup = dict(device=next(model.parameters()).device,
                          dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.batch_size = batch_size
        self.epochs = epochs
        self.idlg = idlg
        self.loss_fn = loss_fn

    def reconstruct(self, gt_gradient, shape=(3, 32, 32), *, plot_fn=None):
        """
        Run the DLG optimization.
        """

        # 1. Initialize the data.
        x_trial = self._init_data(shape)

        # 2. Initialize the label.
        out = self.model(x_trial)
        y_trial = torch.randn(out.shape[1]).to(
            **self.setup).requires_grad_(True)

        # 3. Extract the ground truth label
        # using iDLG trick.
        reconstruct_label = True
        if self.idlg and self.batch_size == 1:
            last_weight_min = torch.argmin(
                torch.sum(gt_gradient[-2], dim=-1), dim=-1)
            y_trial = last_weight_min.detach().reshape((1,)).requires_grad_(False)
            reconstruct_label = False
        else:
            # If we need to reconstruct the label, we need to change the loss fn.
            def loss_fn(pred, labels):
                labels = torch.nn.functional.softmax(labels, dim=-1)
                return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
            self.loss_fn = loss_fn

        # 4. Create the optimizer.
        if reconstruct_label:
            optimizer = torch.optim.LBFGS([x_trial, y_trial],  lr=1,
                                          tolerance_grad=1e-7, tolerance_change=1e-9, max_iter=100,
                                          line_search_fn="strong_wolfe")
        else:
            optimizer = torch.optim.LBFGS([x_trial],  lr=1,
                                          tolerance_grad=1e-7, tolerance_change=1e-9, max_iter=100,
                                          line_search_fn="strong_wolfe")

        epochs = self.epochs
        dm, ds = self.mean_std

        # 5. Run the optimization.
        print("Starting optimization...")
        start = time.time()
        try:
            if plot_fn:
                fig, axes = plot_fn(
                    x_trial.data, fig=None, axes=None, denormalize=True, mean=dm, std=ds)

            for iteration in range(epochs):
                closure = self._gradient_closure(
                    optimizer, x_trial, y_trial, gt_gradient)

                rec_loss = optimizer.step(closure)

                with torch.no_grad():
                    x_trial.data = torch.max(
                        torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == epochs) or iteration % 10 == 0:
                        print(
                            f'It: {iteration}. Rec. loss: {rec_loss.item():2.8f}.')
                    if (iteration % 10 == 0) and plot_fn:
                        fig, axes = plot_fn(
                            x_trial.data, fig, axes, denormalize=True, mean=dm, std=ds)

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass

        elapsed = time.time() - start
        print(f'Elapsed time: {elapsed:.2f} seconds.')

        return x_trial.detach(), y_trial.detach()

    def _init_data(self, shape):
        return torch.randn((self.batch_size, *shape), **self.setup).requires_grad_()

    def _gradient_closure(self, optimizer, x_trial, y_trial, gt_gradient):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            out = self.model(x_trial)
            loss = self.loss_fn(out, y_trial)

            gradient = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True)

            rec_loss = self._reconstruction_costs(gradient, gt_gradient)

            rec_loss.backward()
            return rec_loss
        return closure

    def _reconstruction_costs(self, grad, gt_grad):
        """Input gradient is given data."""
        indices = torch.arange(len(gt_grad))  # Default indices.
        weights = gt_grad[0].new_ones(len(gt_grad))  # Same weight.
        costs = 0
        for i in indices:
            costs += ((grad[i] - gt_grad[i]).pow(2)).sum() * weights[i]
        return costs


if __name__ == '__main__':
    import argparse
    import numpy as np

    def get_device():
        """Returns the appropriate device (CUDA if available, otherwise CPU)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def extract_ground_truth(dataloader, args, setup):
        """Extracts the ground truth data and label from the dataloader."""
        if args.batch_size == 1:
            target_id = np.random.randint(len(dataloader.dataset))
            gt_data, gt_label = dataloader.dataset[target_id]
            gt_data, gt_label = (
                gt_data.unsqueeze(0).to(**setup),
                torch.as_tensor((gt_label,), device=setup["device"]),
            )
            data_shape = (3, gt_data.shape[2], gt_data.shape[3])
        else:
            gt_data, gt_label = [], []
            target_id = np.random.randint(len(dataloader.dataset))
            while len(gt_label) < args.batch_size:
                data, label = dataloader.dataset[target_id]
                target_id += 1
                if label not in gt_label:
                    gt_label.append(torch.as_tensor(
                        (label,), device=setup["device"]))
                    gt_data.append(data.to(**setup))
            gt_data = torch.stack(gt_data)
            gt_label = torch.cat(gt_label)
            data_shape = (3, gt_data.shape[2], gt_data.shape[3])
        return gt_data, gt_label, data_shape

    setup = dict(device=get_device(), dtype=torch.float)

    parser = argparse.ArgumentParser(
        description='Deep Leakage from Gradients (DLG).')

    parser.add_argument('--batch_size', default=1, type=int,
                        help='How many images should be recovered from the given gradient.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='How many epochs to run the optimizer.')
    parser.add_argument('--idlg', default=True, type=bool,
                        help='Use the iDLG trick if batch_size = 1.')

    args = parser.parse_args()

    print(f"Args: {args}")

    num_classes = 100
    loss_fn, trainloader, validloader = load_cifar100(
        batch_size=args.batch_size)

    # print some data
    print(f"Training set size: {len(trainloader.dataset)}")
    print(f"Validation set size: {len(validloader.dataset)}")

    dm = torch.as_tensor(cifar100_mean, **setup)[:, None, None]
    ds = torch.as_tensor(cifar100_std, **setup)[:, None, None]

    model, model_seed = construct_model(num_classes, num_channels=3)
    model.to(**setup)
    model.eval()

    gt_data, gt_label, shape = extract_ground_truth(validloader, args, setup)

    print(f"Ground truth data shape: {gt_data.shape}")
    print(f"Ground truth label shape: {gt_label.shape}")

    display_batch(gt_data, denormalize=True, mean=dm, std=ds)

    # Run reconstruction.
    model.zero_grad()
    target_loss = loss_fn(model(gt_data), gt_label)
    gt_gradient = torch.autograd.grad(target_loss, model.parameters())
    gt_gradient = [grad.detach() for grad in gt_gradient]
    gt_gradnorm = torch.stack([g.norm() for g in gt_gradient]).mean()

    print(f"Full gradient norm is {gt_gradnorm:e}.")

    dlg = DLG(model, loss_fn=loss_fn, mean_std=(dm, ds), batch_size=args.batch_size,
              epochs=args.epochs, idlg=args.idlg)

    data, _ = dlg.reconstruct(
        gt_gradient, shape=shape, plot_fn=display_batch_iterative)

    print("Reconstruction finished.")

    display_batch(data, denormalize=True, mean=dm, std=ds)
