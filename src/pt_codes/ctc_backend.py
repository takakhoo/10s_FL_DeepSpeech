import torch
import torch.nn as nn

def train_with_ctc(trial, device):
    
    # Random seed
    torch.manual_seed(0)

    # Data
    torch.manual_seed(0)
    inp=torch.randn(500,16,10, device=device)    
    targets = torch.randint(1, 20, (sum(range(100,116)),), dtype=torch.int)
    input_lengths = torch.full((16,), 500, dtype=torch.long)
    target_lengths = torch.arange(100,116, dtype=torch.long)
    
    # Network
    net=nn.Sequential(nn.Linear(10,20), torch.nn.LogSoftmax(2))
    net.to(device)
    optimizer=torch.optim.Adam(net.parameters())
    ctc_loss = nn.CTCLoss(blank=19) # use cuDNN: blank=0, use CUDA: blank !=0 (only true when other parameters are not changed)
        
    # Training loop
    inc_loss=0
    for it in range(100):
        optimizer.zero_grad()
        log_probs=net(inp)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        inc_loss+=float(loss)

    # Print
    torch.cuda.synchronize()
    net_csum=torch.nn.utils.parameters_to_vector(net.parameters()).sum()
    print('Trial {} ::: Device {:4} ::: Iter {} ::: net_csum {:.10f} ::: log_prob_checksum {:.10f} ::: loss {:.10f} ::: inc_loss {:.10f}'.format(trial, device, it, net_csum,log_probs.sum(), float(loss), inc_loss))

if __name__ == '__main__':

    # Get cuDNN version
    print(torch.backends.cudnn.version())
    
    # Toggle cuDNN backend    
    # torch.backends.cudnn.deterministic = True # change to True and blank label to 0 to toggle CUDNN_CTC_LOSS_ALGO_DETERMINISTIC / CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC in pytorch/aten/src/ATen/native/cudnn/LossCTC.cpp
    # torch.backends.cudnn.benchmark = False
    
    # Train with CTC
    for device in ['cuda', 'cpu']:
        for trial in range(3):
            train_with_ctc(trial, device)