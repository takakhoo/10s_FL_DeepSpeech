# %%
import torch
import torchaudio
from IPython.display import Audio
from torch.utils.data import DataLoader
import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter


#


# %%
class CustomAudioDataset(Dataset):
    def __init__(self, tsv_path, clips_dir, config):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.clips_dir = clips_dir
        self.config = config  # Configurations for MFCC transformation
        self.char_to_num = config['char_to_num']  # Character to number mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx]['path'] + '.mp3'
        label_str = self.data.iloc[idx]['sentence']

        # Convert label string to numbers using char_to_num mapping
        label_nums = torch.tensor([self.char_to_num[c] for c in label_str if c in self.char_to_num])

        audio_file = os.path.join(self.clips_dir, filename)
        waveform, sample_rate = torchaudio.load(audio_file)

        # Perform MFCC transformation
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=self.config['mfcc_bins'],
            melkwargs={"n_fft": int(sample_rate * self.config['win_length_ms'] // 1000),
                       "hop_length": int(sample_rate * self.config['step_length_ms'] // 1000),
                       "n_mels": self.config['mfcc_bins']}
        )
        mfcc_features = mfcc_transform(waveform).squeeze().T  # [t x f]
        seq_length = mfcc_features.size(0)  # Number of MFCC time steps
        target_length = len(label_nums)  # Length of label_nums


        #return mfcc_features, label_nums, os.path.join(self.clips_dir, filename), label_str
        return mfcc_features, label_nums, seq_length, target_length





# %%
# Define collate function for padding sequences
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # Sort the batch by sequence length (number of frames)
    #batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
    
    # Get the sequences and labels from the sorted batch
    sequences, labels, seq_lengths, target_lengths = zip(*batch)
    
    # Pad sequences with zeros to make them equal in length (number of frames)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels           = pad_sequence(labels, batch_first=True)
    
    return padded_sequences, labels, seq_lengths, target_lengths

# %%
# Example usage
# tsv_path = '/scratch/f006pq6/datasets/commonvoice_v2/train.tsv'
# clips_dir = '/scratch/f006pq6/datasets/commonvoice_v2/clips'

# dataset = CustomAudioDataset(tsv_path, clips_dir,config)
# sample = dataset[0]  # Get the first sample
# mfcc, label, path, label_str = sample

# # print("Waveform shape:", waveform.shape)
# # print("mfcc shape:", mfcc.shape)
# # print("Label:", label)

# dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True)

# # Example usage of the data loader
# # for batch in dataloader:
# #     padded_sequences, labels, paths, label_str = batch
# #     print("Padded Sequences Shape:", padded_sequences.shape)
# #     print("Labels:", labels)
# #     print("Paths:", paths)
# #     break  # Break after processing the first batch

# %%
class DeepSpeech1(nn.Module):
    def __init__(self, config):
        super(DeepSpeech1, self).__init__()

        self.config = config

        # Create layers
        #self.layer_1 = nn.Linear(config['n_input'] + 2 * config['n_input'] * config['n_context'], config['n_hidden_1'])
        self.layer_1 = nn.Conv1d(config['n_input'], config['n_hidden_1'],
                                 kernel_size=2 * config['n_context']+1, stride =1,
                                 padding=config['n_context'])
        self.layer_2 = nn.Linear(config['n_hidden_1'], config['n_hidden_2'])
        self.layer_3 = nn.Linear(config['n_hidden_2'], config['n_hidden_3'])
        self.layer_5 = nn.Linear(config['n_cell_dim'], config['n_hidden_5'])
        self.layer_6 = nn.Linear(config['n_hidden_5'], config['n_hidden_6'],bias=False)

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=config['n_hidden_3'], hidden_size=config['n_cell_dim'], batch_first=True)


    def forward(self, batch_x, seq_length, previous_state=None, reuse=False):
        #batch x : [B, T,  F]
        batch_x = batch_x.transpose(1,2) # [B, F , T]
        B       = batch_x.shape[0]


        # Pass through layers with clipped ReLU activation and dropout
        layer_1 = self.layer_1(batch_x) # [B, n_hidden, T]
        layer_1 = layer_1.transpose(1,2)  # [B, T, n_hidden]
        layer_1 = layer_1.reshape(-1, self.config['n_hidden_1'])
        layer_1 = F.dropout(F.relu(layer_1), p=self.config['dropout'][0])
        layer_2 = F.dropout(F.relu(self.layer_2(layer_1)), p=self.config['dropout'][1])
        layer_3 = F.dropout(F.relu(self.layer_3(layer_2)), p=self.config['dropout'][2])

        # Reshape output for LSTM input
        layer_3 = layer_3.reshape(B, -1, self.config['n_hidden_3']) #[B, T, F]

        # Run through LSTM
        # lstm_output, lstm_output_state = self.rnn_impl(layer_3, seq_length, previous_state, reuse)
        # Run through LSTM
        lstm_output, lstm_output_state = self.lstm(layer_3, 
                                                   None if not previous_state else previous_state)
        # Pass through remaining layers
        lstm_output = lstm_output.reshape(-1, self.config['n_cell_dim'])  # [n_steps*batch_size, n_cell_dim]
        layer_5 = F.dropout(F.relu(self.layer_5(lstm_output)), p=self.config['dropout'][5])
        raw_logits = self.layer_6(layer_5).reshape(B, -1, self.config['n_hidden_6'])

        return raw_logits

class DeepSpeech1Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepSpeech1(config)
        self.loss_fn = nn.CTCLoss()

    def forward(self, batch_x, seq_length, previous_state=None, reuse=False):
        return self.model(batch_x, seq_length, previous_state, reuse)

    def training_step(self, batch, batch_idx):
        padded_sequences, labels, seq_lengths, target_lengths = batch
        seq_length = torch.sum(padded_sequences.sum(dim=2) != 0, dim=1)  # Calculate sequence length

        logits = self(padded_sequences, seq_length)
        logits = logits.permute(1, 0, 2) # Shape: (time_step, batch_size, 29)
        logits = F.log_softmax(logits, dim = -1)
        loss   = self.loss_fn(logits, labels, seq_lengths, target_lengths )
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def train_dataloader(self):
        return DataLoader(dataset, batch_size=self.config['batch_size'], collate_fn=collate_fn, shuffle=True,
                           pin_memory=True, num_workers=config['cpus'])       

#%%
if __name__ == '__main__':
    #data config
    config = {
        'sampling_rate': 16000,
        'win_length_ms': 32,
        'step_length_ms': 20,
        'mfcc_bins': 26,
        'batch_size': 32
    }

    # Define the character-to-number mapping with space as 0 and ' as the last number
    char_to_num = {
        ' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
        'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18,
        's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, "'": 27
    }
    num_to_char = {v: k for k, v in char_to_num.items()}

    config['char_to_num'] = char_to_num
    config['num_to_char'] = num_to_char

    #network config
    # Update the existing config dictionary with the specified values
    config.update({
        'n_hidden_1': 2048,
        'n_hidden_2': 2048,
        'n_hidden_3': 2048,
        'n_hidden_5': 2048,
        'n_hidden_6': len(char_to_num)+1,
        'n_cell_dim': 2048,
        'dropout': [0.0] * 6,  # Assuming 6 dropout rates are needed
        'overlap': True,
        'layer_norm': False,
        'n_input': 26,
        'n_context': 9
    })

    # training config
    config.update({
        'gpus': 4,
        'learning_rate':0.1,
        'cpus':16,
        'max_epoch': 100,
        'half_precision': False,  # Enable half precision

    })


    # %%
    # Initialize your dataset and model
    tsv_path = '/scratch/f006pq6/datasets/commonvoice_v2/train.tsv'
    clips_dir = '/scratch/f006pq6/datasets/commonvoice_v2/clips'
    dataset = CustomAudioDataset(tsv_path, clips_dir, config)

    model = DeepSpeech1Model(config)

    # Set up TensorBoard logger
    tb_logger = pl.loggers.TensorBoardLogger('logs/', name='DeepSpeech1_experiment')
    tb_logger.log_hyperparams(config)

    # Set up model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='DeepSpeech1_{epoch:02d}',
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=10,  # Save every 10 epochs
    )

    # Initialize PyTorch Lightning Trainer
    #trainer = pl.Trainer(fast_dev_run=True)
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epoch'],
        gpus=config['gpus'],  # Let PyTorch Lightning decide based on available resources
        logger=tb_logger,  # Use TensorBoard logger
        callbacks=[checkpoint_callback],  # Add model checkpoint callback
        precision=16 if config['half_precision'] else 32, 
        accelerator='ddp',  # Use DDP with multiple GPUs if available, else use single GPU or CPU
    )
    # Start training
    trainer.fit(model)