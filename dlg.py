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

import math
from typing import Iterable
import torch

class DLG:
    def __init__(self, model, loss, grad, input_size, output_size, batch_size, device, step_size=0.1):
        """
        A simple DLG optimizer with a fixed step size.

        Parameters:
        - model: The model to attack.
        - loss: The loss function of the model.
        - grad: The gradient function of the model.
        - input_size: The size of the input data.
        - output_size: The size of the output data.
        - batch_size: The batch size for the attack.
        - step_size: The fixed step size for updates.
        """
        self.model = model
        self.loss = loss
        self.grad = grad
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.step_size = step_size

        self.target_data, self.target_label = self._init(device)

    def optimize(self, gt, *, max_iter=100, print_it=10, plot_fn=None,
                 plot_it=50):
        """
        Run the DLG optimization.

        Parameters:
        - bs: The batch size approximation.
        - gt: The ground-truth gradients.
        - max_iter: Maximum number of iterations.
        - print_it: Print the loss every `print_it` iterations.
        - plot_it: Plot the images every `plot_it` iterations
        - plot_fn: A function to call at each iteration.

        Returns:
        - history: A list of data at each iteration.
        """

        optimizer = torch.optim.LBFGS([self.target_data, self.target_label],  lr=1,
                                      tolerance_grad=1e-7, tolerance_change=1e-9, max_iter=100,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            g = self._grad()

            grad_diff = sum(((gx - gy) ** 2).sum()
                            for gx, gy in zip(g, gt))

            grad_diff.backward()

            return grad_diff

        if plot_fn:
            fig, axes = plot_fn(self.get_target())

        for iters in range(epochs):
            optimizer.step(closure)
            if iters % print_it == 0:
                current_loss = closure().item()
                print(f"Iteration {iters}: Loss = {current_loss:.8f}")
            if iters % plot_it == 0 and plot_fn:
                fig, axes = plot_fn(self.get_target(), fig,  # pyright: ignore
                                    axes)  # pyright: ignore
            if iters == max_iter:
                break

    def get_target(self):
        """Returns pairs of (data, label) as required by the interface."""
        # Calculate the number of elements in each data and label
        # Total number of elements in one data point
        data_elements = torch.prod(torch.tensor(self.input_size))
        # Total number of elements in one label point
        label_elements = torch.prod(torch.tensor(self.output_size))

        # Iterate over the flattened data and labels, reshaping them back to their original shapes
        data = [self.target_data[i:i + data_elements].view(self.input_size)
                for i in range(0, len(self.target_data), data_elements.item())]  # pyright: ignore
        label = [self.target_label[i:i + label_elements].view(self.output_size)
                 for i in range(0, len(self.target_label), label_elements.item())]  # pyright: ignore

        return [(d, l) for d, l in zip(data, label)]

    def _grad(self):
        """Computes the gradient of the target image."""
        total_loss = torch.tensor(0., device=device, requires_grad=True)
        for data, label in self.get_target():
            assert data.requires_grad, "Data tensor must require gradients!"
            assert label.requires_grad, "Label tensor must require gradients!"
            pred = self.model(data)
            total_loss = total_loss + self.loss(pred, label)
        assert total_loss.requires_grad, "Loss tensor must require gradients!"
        gradients = self.grad(
            total_loss, model.parameters(), create_graph=True)
        return gradients

    def _init(self, device) -> Iterable[torch.Tensor]:
        """Generates param groups with random data and labels."""
        all_data = []
        all_labels = []

        for _ in range(self.batch_size):
            dummy_data = torch.randn(
                self.input_size, device=device, requires_grad=True)
            dummy_label = torch.randn(
                self.output_size, device=device, requires_grad=True)

            all_data.append(dummy_data.view(-1))  # Flatten data
            all_labels.append(dummy_label.view(-1))  # Flatten label

        target_data = torch.cat(all_data, dim=0)
        target_labels = torch.cat(all_labels, dim=0)

        return torch.nn.Parameter(target_data), torch.nn.Parameter(target_labels)


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.autograd import grad
    from vision.resnet import LeNet, weights_init
    from utils import cross_entropy_for_onehot, label_to_onehot, cross_entropy_for_onehot

    print(torch.__version__, torchvision.__version__)

    def get_device():
        """Returns the appropriate device (CUDA if available, otherwise CPU)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_dataset():
        """Loads the CIFAR-100 dataset."""
        return datasets.CIFAR100("~/.torch", download=True)

    def preprocess_image(image, device):
        """Converts an image to a tensor and moves it to the given device."""
        return transforms.ToTensor()(image).to(device)

    def display_batch_iterative(data, fig=None, axes=None):
        """Updates images in a single window dynamically without opening multiple tabs.

        Args:
            data: List of tuples (image, label) containing the target images.
            fig: (optional) Matplotlib figure to reuse.
            axes: (optional) Matplotlib axes to reuse.

        Returns:
            fig, axes: The figure and axes for reuse.
        """
        max_per_row = 10
        total_images = len(data)
        ncols = min(max_per_row, total_images)
        nrows = math.ceil(total_images / max_per_row)

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

        for idx, (img, _) in enumerate(data):
            row = idx // max_per_row
            col = idx % max_per_row
            ax = axes[row][col]
            pil_img = transforms.ToPILImage()(
                img[0].cpu())  # Convert tensor to PIL Image
            ax.imshow(pil_img)
            ax.axis('off')

        # Hide any unused axes if total_images doesn't fill the grid completely
        for idx in range(total_images, nrows * ncols):
            row = idx // max_per_row
            col = idx % max_per_row
            axes[row][col].axis('off')

        fig.canvas.draw()  # Redraw the figure
        fig.canvas.flush_events()  # Process UI events

        return fig, axes  # Return the figure and axes for reuse

    def display_batch(data):
        """Displays the target images in a grid.

        Args:
            data: List of tuples (image, label) containing the target images.
        """
        max_per_row = 10
        total_images = len(data)
        ncols = min(max_per_row, total_images)
        nrows = math.ceil(total_images / max_per_row)
        plt.figure(figsize=(ncols * 1.5, nrows * 1.5))

        for idx, (img, _) in enumerate(data):
            row = idx // max_per_row
            col = idx % max_per_row
            plt.subplot(nrows, ncols, idx + 1)
            pil_img = transforms.ToPILImage()(img[0].cpu())
            plt.imshow(pil_img)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def load_batch(batch_size, dataset, device):
        """Loads and preprocesses the target image and label."""
        assert 0 < batch_size <= len(dataset), "batch_size out of bounds."

        gt_data = []
        for _ in range(batch_size):
            img_index = torch.randint(len(dataset), size=(1,)).item()
            gt_input = preprocess_image(dataset[img_index][0], device)
            gt_input = gt_input.view(1, *gt_input.size())
            gt_label = torch.tensor([dataset[img_index][1]],
                                    dtype=torch.long, device=device)
            gt_onehot_label = label_to_onehot(gt_label)
            gt_data.append((gt_input, gt_onehot_label))

        return gt_data

    def initialize_model(device):
        """Initializes the LeNet model and applies weight initialization."""
        torch.manual_seed(1234)
        net = LeNet().to(device)
        net.apply(weights_init)
        return net

    def compute_gradients(model, batch, device):
        """Computes the original gradient of the target image.
        Args:
            model: The model used for inference.
            data: A batch that is a List of tuples (image, label).
        Returns:
            the averaged gradient of the target image.
        """
        total_loss = torch.tensor(0., device=device)
        for data, label in batch:
            pred = model(data)
            total_loss += cross_entropy_for_onehot(pred, label)
        gradients = grad(total_loss, model.parameters())
        return [g.detach().clone() for g in gradients]

    def initialize_target(input_size, output_size, batch_size, device):
        """Generates random dummy data and labels."""
        batch = []
        for _ in range(batch_size):
            dummy_data = torch.randn(
                input_size, device=device, requires_grad=True)
            dummy_label = torch.randn(
                output_size, device=device, requires_grad=True)
            batch.append((dummy_data, dummy_label))
        return batch

    def loss_fn(pred, label):
        target_onehot_label = F.softmax(label, dim=-1)
        target_loss = cross_entropy_for_onehot(
            pred, target_onehot_label)
        return target_loss

    def grad_fn(loss, model_params, create_graph=True):
        return grad(loss, model_params, create_graph=create_graph)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Deep Leakage from Gradients.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of target images to extract from CIFAR-100.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of optimization iterations.')
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    device = get_device()
    print(f"Running on {device}")

    print("Loading CIFAR-100 dataset...")
    dataset = load_dataset()
    print("Dataset loaded.")

    print("Extracting {} target images...".format(args.batch_size))
    batch = load_batch(args.batch_size, dataset, device)
    display_batch(batch)
    print("Target images loaded.")

    print("Initializing model...")
    model = initialize_model(device)
    print("Model initialized.")

    print("Initialize attack target...")
    input_size = batch[0][0].shape
    output_size = batch[0][1].shape
    print("Attack target initialized.")

    print("Computing ground truth gradients...")
    gt = compute_gradients(model, batch, device)
    print("Truth gradients computed.")
    print("Optimizing...")
    dlg = DLG(model, loss_fn, grad_fn, input_size,
              output_size, batch_size, device)
    display_batch(dlg.get_target())
    dlg.optimize(gt, max_iter=epochs, plot_fn=display_batch_iterative)
    display_batch(dlg.get_target())
    print("Done.")
