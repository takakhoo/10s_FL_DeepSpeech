import torch 

def ctc_loss_imp(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        # ==========================================================================================================
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        # ==========================================================================================================
        probs = log_probs[:input_length, i].exp()
        # ==========================================================================================================
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        # ==========================================================================================================
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output
    
 
from typing import Any, Tuple
import torch
import random
import numpy as np
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


reduction = 'mean'
BLANK     = 0
neginf    = torch.tensor(-float('inf'))

def ctc_loss_imp_not_parallel(inp,  targets, inp_len, tgt_len):
    # inp in log space
    inp_len       = torch.as_tensor(inp_len, dtype=torch.int)
    tgt_len       = torch.as_tensor(tgt_len, dtype=torch.int)
    cum_tgt_len   = tgt_len.cumsum(0)
    losses        = []
    alpha_global  = []

    for i in range(inp.size(1)):
        inp_length          = inp_len[i].item()
        target_length       = tgt_len[i].item()
        cum_target_length   = cum_tgt_len[i].item()

        targets_prime = targets.new_full((2 * target_length + 1,), BLANK)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]

        probs = inp[:inp_length, i]
        alpha   = inp.new_ones((inp_length, target_length*2+1)) * neginf 
        alpha[0,0] = probs[0, BLANK]
        if target_length > 0:
            alpha[0,1] = probs[0, targets_prime[1]]

        for t in range(1, inp_length):
            for s in range(2 * target_length +1):
                a1 = alpha[t-1,s]
                a2 = alpha[t-1,s-1] if s > 0 else neginf
                a3 = alpha[t-1,s-2] if s > 1 and targets_prime[s-2]!= targets_prime[s] else neginf

                amax = max(a1,a2,a3)
                amax = 0 if amax == neginf else amax
                
                alpha[t,s] = torch.log( torch.exp(a1-amax) + torch.exp(a2-amax) + torch.exp(a3-amax)) + \
                    amax + probs[t, targets_prime[s]]

        if target_length == 0:
            loss = -alpha[-1,0]
        else:
            l1 = alpha[-1,-2]
            l2 = alpha[-1,-1]
            loss = -torch.log(torch.exp(l1)+ torch.exp(l2))
        losses.append(loss[None])
        alpha_global.append(alpha)
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        print( (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean() )
    elif reduction == 'sum':
        print( output.sum() )

    output_mean = (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean()

    return output_mean

neginf    = torch.tensor(-float('inf'))
def ctc_loss_imp_logspace(inp, targets, inp_len, tgt_len, BLANK=0, reduction='mean'):
        # inp in log space
        inp_len       = torch.as_tensor(inp_len, dtype=torch.int)
        tgt_len       = torch.as_tensor(tgt_len, dtype=torch.int)
        cum_tgt_len   = tgt_len.cumsum(0)
        losses        = []
        alpha_global  = []

        for i in range(inp.size(1)):
            inp_length          = inp_len[i].item()
            target_length       = tgt_len[i].item()
            cum_target_length   = cum_tgt_len[i].item()

            targets_prime = targets.new_full((2 * target_length + 1,), BLANK)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, :target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]

            probs = inp[:inp_length, i]
            alpha   = inp.new_ones((inp_length, target_length*2+1)) * neginf 
            alpha[0,0] = probs[0, BLANK]
            if target_length > 0:
                alpha[0,1] = probs[0, targets_prime[1]]

            for t in range(1, inp_length):
                for s in range(2 * target_length +1):
                    a1 = alpha[t-1,s]
                    a2 = alpha[t-1,s-1] if s > 0 else neginf
                    a3 = alpha[t-1,s-2] if s > 1 and targets_prime[s-2]!= targets_prime[s] else neginf

                    amax = max(a1,a2,a3)
                    amax = 0 if amax == neginf else amax
                    
                    alpha[t,s] = torch.log( torch.exp(a1-amax) + torch.exp(a2-amax) + torch.exp(a3-amax)) + \
                        amax + probs[t, targets_prime[s]]

            if target_length == 0:
                loss = -alpha[-1,0]
            else:
                l1 = alpha[-1,-2]
                l2 = alpha[-1,-1]
                loss = -torch.log(torch.exp(l1)+ torch.exp(l2))
            losses.append(loss[None])
            alpha_global.append(alpha)
        output = torch.cat(losses, 0)
        if reduction == 'mean':
            print( (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean() )
        elif reduction == 'sum':
            print( output.sum() )

        output_mean = (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean()

        return output_mean