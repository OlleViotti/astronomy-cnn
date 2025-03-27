from huggingface_hub import hf_hub_download
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.preprocessing import QuantileTransformer

data_dir = "./data"
np.random.seed(42)
torch.manual_seed(42)


def load_data(data_dir):
    "Load data from file or from hugging face"
    try:
        spectra = np.load(f"{data_dir}/spectra.npy")
    except FileNotFoundError:
        hf_hub_download(
            repo_id="simbaswe/galah4",
            filename="spectra.npy",
            repo_type="dataset",
            local_dir=data_dir,
        )
        spectra = np.load(f"{data_dir}/spectra.npy")

    try:
        labels = np.load(f"{data_dir}/labels.npy")
    except FileNotFoundError:
        hf_hub_download(
            repo_id="simbaswe/galah4",
            filename="labels.npy",
            repo_type="dataset",
            local_dir=data_dir,
        )
        labels = np.load(f"{data_dir}/labels.npy")
    spectra = torch.tensor(spectra).float()
    labels = torch.tensor(labels).float()
    return spectra, labels


def normalize_spectra(spectra):
    "Normalize spectra using logarithm"
    spectra_norm = torch.log(torch.clip(spectra, min=0.2))
    return spectra_norm


def normalize_labels(labels):
    "Normalize to zero mean and unit variance"
    mean = labels.mean(dim=0)
    std = labels.std(dim=0)
    return (labels - mean) / std


def subset_labels(labels, label_names):
    all_label_names = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
    boolean_mask = [l in label_names for l in all_label_names]
    labels = labels[:, boolean_mask]
    return labels, label_names

def plot_spectra(spectra, labels, label_names, idx=0):
    "Plot one or several spectra as a line"
    fig, ax = plt.subplots()

    ax.plot(spectra[idx, :])
    ax.set_xlabel("Wavelength [index]")
    ax.set_ylabel("Flux [normalized]")
    label_values = labels[idx]
    info = '\n'.join([f'{n}={v:.4g}' for n,v in list(zip(label_names, label_values))])
    ax.set_title(f'Spectra {idx}')
    ax.text(0.99, 0.01, info, transform=ax.transAxes,
        ha='right', va='bottom')

    fig.tight_layout()
    fig.savefig(f"./plots/spectra_{idx}.pdf", dpi=600)


def plot_spectra_imshow(spectra, idx=range(100)):
    "Plot several spectra as a grid"
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(spectra[idx, :], cmap="viridis", aspect="auto")
    ax.set_xlabel("Wavelength [index]")
    ax.set_ylabel("Spectra [index]")
    fig.colorbar(im, label="Flux [normalized]")
    fig.tight_layout()
    fig.savefig("./plots/spectra_imshow_plot.pdf", dpi=600)

def plot_label_hist(plot_data, label_names, suffix=''):
    num_labels=len(label_names)
    ncols = int(np.ceil(np.sqrt(num_labels)))
    nrows = int(np.ceil(num_labels / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))

    for i, ax in enumerate(axs.flatten()):
        if i < num_labels:
            ax.hist(plot_data[:, i], bins=50)
            ax.set_xlabel(f'{label_names[i]} [{suffix}]')
            ax.set_ylabel('count')
        else:
            ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(f'./plots/labels_hist_{suffix}.pdf', dpi=600)  

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # CUDA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # if nothing is found use the CPU
    return device


def get_num_workers(save_n=0):
    return os.cpu_count() - save_n


def get_data(device=None, label_names=["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]):
    spectra, labels = load_data(data_dir)
    spectra_norm = normalize_spectra(spectra)
    labels, label_names = subset_labels(labels, label_names)
    # This commented out code is good for some of the other labels that we do not consider here.
    # qt = QuantileTransformer(output_distribution='normal', random_state=0)
    # labels_norm = torch.tensor(qt.fit_transform(labels)).float()
    labels_norm = normalize_labels(labels)  # Normalize to zero mean and unit variance
    device = device if device is not None else get_device()
    spectra, labels, spectra_norm, labels_norm = [tensor.to(device) for tensor in [spectra, labels, spectra_norm, labels_norm]]
    return spectra, labels, spectra_norm, labels_norm, label_names

def generate_plots():
    spectra, labels, spectra_norm, labels_norm, label_names = get_data('cpu')

    # Generate plots for 5 random spectra
    for i in np.random.choice(spectra.shape[0], 5):
        plot_spectra(spectra_norm, labels, label_names, i)
    plot_spectra_imshow(spectra_norm)
    plot_label_hist(labels_norm, label_names, suffix='normalized')
    plot_label_hist(labels, label_names)
    

if __name__ == "__main__":
    generate_plots()
