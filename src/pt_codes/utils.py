# plot utilities

import matplotlib.pyplot as plt
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