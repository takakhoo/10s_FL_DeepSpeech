
import matplotlib.pyplot as plt
import torch
def plot_loss_over_epoch(loss_history,title,x_title,y_title):
    """
    Plot loss values over epochs.
    
    Args:
        loss_history (list): List of loss values for each epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, linestyle='-')
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(True)
    plt.show()

def plot_spectrogram(tensor):
    """
    Plot a spectrogram from a PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
    """
    # Ensure that the input tensor is on the CPU and in the numpy format
    spectrogram = tensor.squeeze().cpu().numpy()
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, cmap='viridis', origin='lower', aspect='auto')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_four_graphs(gt_tensor, reconstructed_tensor, loss_array,epoch=0):
    """
    Plot four graphs: ground truth spectrogram, reconstructed spectrogram, 
    difference between the two, and loss over epoch.
    
    Args:
        gt_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, channels, height, width).
        reconstructed_tensor (torch.Tensor): Reconstructed tensor of same shape as `gt_tensor`.
        loss_array (List[float]): Array of loss values over epochs.
    """
    diff_tensor = torch.abs(gt_tensor - reconstructed_tensor)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Ground Truth Spectrogram
    axs[0, 0].imshow(gt_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
    axs[0, 0].set_title('Ground Truth Spectrogram')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Frequency')

    # Reconstructed Spectrogram
    axs[0, 1].imshow(reconstructed_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
    axs[0, 1].set_title('Reconstructed Spectrogram')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Frequency')
    
    # Difference Spectrogram
    axs[1, 0].imshow(diff_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
    axs[1, 0].set_title('Difference Spectrogram')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frequency')

    # Loss over epoch
    axs[1, 1].plot(loss_array)
    axs[1, 1].set_title('Loss Over Epochs')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    # save figure with name that has date time hour min and epoch number
    import datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M")
    plt.savefig('figures/{}_{}.png'.format(now, epoch))
    plt.show()

    
