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

class CustomCTCLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, inp_len, tgt_len, targets):
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

        ctx.save_for_backward(inp,inp_len, tgt_len, targets,torch.stack(alpha_global,dim=0), torch.tensor(losses))
        return output_mean
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inps, inp_lens, tgt_lens, targets, alphas, losses = ctx.saved_tensors
        grad_inp = torch.ones_like(inps) * neginf

        # compute beta  
        cum_tgt_len = tgt_lens.cumsum(0)

        dt = inps.dtype

        # lp_to_l = lambda idx,tgt: 0 if idx%2==0 else tgt[idx//2] # label prime -> label

        for i in range(inps.size(1)): # loop through each example in dataset
            inp_length        = inp_lens[i].item()
            target_length     = tgt_lens[i].item()
            cum_target_length = cum_tgt_len[i].item()

            # ========================================================================================================= = 
            # to do: remove this target prime array, not needed.
            # use the lp_to_l later.
            targets_prime = targets.new_full((2 * target_length + 1,), BLANK)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, :target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
            # ==========================================================================================================
            probs = inps[:inp_length, i]
            # ==========================================================================================================
            alpha         = alphas[i]
            beta          = inps.new_ones((inp_length, target_length * 2 + 1)) * neginf

            beta    [-1, -1]       = probs[-1, BLANK]
            grad_inp[-1, i, BLANK] = alpha[-1, -1] + beta[-1, -1]

            if target_length > 0: 
                beta[-1, -2] = probs[-1,targets_prime[-2] ] # tricky
                grad_inp[-1,i,targets_prime[-2]] = alpha[-1, -2] + beta[-1, -2]

            for t in reversed(range(0, inp_length-1)):
                for s in reversed(range(0, 2*target_length+1)):
                    b1 = beta[t+1,s]
                    b2 = beta[t+1,s+1] if s < 2 * target_length else neginf
                    b3 = beta[t+1,s+2] if s < 2 * target_length -1 and targets_prime[s] != targets_prime[s+2] else  neginf

                    beta[t,s] = torch.log(torch.exp(b1)+torch.exp(b2)+torch.exp(b3)) + probs[t,targets_prime[s]]

                    alpha_beta = alpha[t,s] + beta[t,s]
                    if grad_inp[t,i,targets_prime[s]] == neginf:
                        grad_inp[t,i,targets_prime[s]] = alpha_beta
                    else:
                        grad_inp[t,i,targets_prime[s]] = torch.log(torch.exp(alpha_beta) + torch.exp(grad_inp[t,i,targets_prime[s]])) # t,n,c

            # make sure beta is correct by computing loss using beta
            # loss = -torch.log(torch.exp(beta[0,0]) + torch.exp(beta[0,1]))
            # assert torch.allclose(loss, losses[i])

            #-a - b - c - d target prime
	        #-abc target
            # done beta
            # print(beta.shape) 

            # return -torch.div(grad_inp, torch.pow(inps,2)) * torch.exp(losses), None, None, None
            # grad_inp = grad_inp.exp()
            # inps     = inps.exp()

            # loss = -log(p)
            # 1/y^2 . log(y)  z=log(y),   1/(e^2z)

            nll = losses[i]
            gr  = tgt_lens[i]
            for t in range(inp_length):
                for c in range(inps.shape[-1]):
                    # grad_inp[t,i,c] = - 1/(torch.exp(-losses[i])) *  torch.exp(-2*inps[t,i,c]) * torch.exp(grad_inp[t,i,c])     #-1/(inps[t,i,c]**2) *  grad_inp[t,i,c] * torch.exp(losses[i])
                    grad_inp[t,i,c] = (torch.exp(inps[t,i,c]) - torch.exp(grad_inp[t,i,c] + nll - inps[t,i,c]))/gr

        grad_inp = torch.nan_to_num(grad_inp, neginf=0.0) / inps.size(1)
        return grad_inp,  None, None, None


# class CustomCTCWeightMatchLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, grad, fl_grad):
#         '''
#         compute squared L2 distance between model grad & fl_grad

#         '''
#         return torch.norm(grad-fl_grad)**2
#     @staticmethod
#     def backward(ctx: Any, *grad_outputs: Any) -> Any:  

