import torch
from torch import Tensor

reduction = 'mean'
BLANK     = 0
neginf    = torch.tensor(-float('inf'))

class CTCLossGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, alphas, losses):
        # log_probs, input_lengths, target_lengths, targets, alphas, losses = ctx.saved_tensors
        grad_inp = torch.ones_like(log_probs) * neginf

        # compute beta

        cum_tgt_len = target_lengths.cumsum(0)

        dt = log_probs.dtype

        # lp_to_l = lambda idx,tgt: 0 if idx%2==0 else tgt[idx//2] # label prime -> label

        for i in range(log_probs.size(1)): # loop through each example in dataset
            inp_length        = input_lengths[i].item()
            target_length     = target_lengths[i].item()
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
            lprob = log_probs[:inp_length, i]
            # ==========================================================================================================
            alpha         = alphas[i]
            beta          = log_probs.new_ones((inp_length, target_length * 2 + 1)) * neginf

            beta    [-1, -1]       = lprob[-1, BLANK]
            grad_inp[-1, i, BLANK] = alpha[-1, -1] + beta[-1, -1]

            if target_length > 0: 
                beta[-1, -2] = lprob[-1,targets_prime[-2] ] # tricky
                grad_inp[-1,i,targets_prime[-2]] = alpha[-1, -2] + beta[-1, -2]

            for t in reversed(range(0, inp_length-1)):
                for s in reversed(range(0, 2*target_length+1)):
                    b1 = beta[t+1,s]
                    b2 = beta[t+1,s+1] if s < 2 * target_length else neginf
                    b3 = beta[t+1,s+2] if s < 2 * target_length -1 and targets_prime[s] != targets_prime[s+2] else  neginf

                    beta[t,s] = torch.log(torch.exp(b1)+torch.exp(b2)+torch.exp(b3)) + lprob[t,targets_prime[s]]

                    alpha_beta = alpha[t,s] + beta[t,s]
                    if grad_inp[t,i,targets_prime[s]] == neginf:
                        grad_inp[t,i,targets_prime[s]] = alpha_beta
                    else:
                        grad_inp[t,i,targets_prime[s]] = torch.log(torch.exp(alpha_beta) + torch.exp(grad_inp[t,i,targets_prime[s]])) # t,n,c

            nll = losses[i]
            gr  = target_lengths[i]
            for t in range(inp_length):
                for c in range(log_probs.shape[-1]):
                    # grad_inp[t,i,c] = - 1/(torch.exp(-losses[i])) *  torch.exp(-2*inps[t,i,c]) * torch.exp(grad_inp[t,i,c])     #-1/(inps[t,i,c]**2) *  grad_inp[t,i,c] * torch.exp(losses[i])
                    grad_inp[t,i,c] = (torch.exp(log_probs[t,i,c]) - torch.exp(grad_inp[t,i,c] + nll - log_probs[t,i,c]))/gr

        grad_inp = torch.nan_to_num(grad_inp, neginf=0.0) / log_probs.size(1)

        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths)
        return grad_inp

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths = ctx.saved_tensors
        cum_tgt_len   = target_lengths.cumsum(0)

        # temporary rebuild everything without using precomputed alpha betas
        # so need to compute alpha, beta, and dalpha/dinput, dbeta/dinput

        # dadi[i,j,k,l] = derivative of alpha[i,j] w.r.t input[k,l]
        # dbdi[i,j,k,l] = derivative of beta[i,j] w.r.t input[k,l]

        
        for i in range(log_probs.size(1)):
            inp_length          = input_lengths[i].item()
            target_length       = target_lengths[i].item()
            cum_target_length   = cum_tgt_len[i].item()

            targets_prime = targets.new_full((2 * target_length + 1,), BLANK)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, :target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]


            lprob = log_probs[:inp_length, i]

            #------------- construct alphas & dadi
            dadi    = log_probs.new_zeros((inp_length, target_length*2+1,  lprob.shape[0], lprob.shape[1]))
            dada    = log_probs.new_zeros((inp_length, target_length*2+1, inp_length, target_length*2+1 ))
            
            alpha   = log_probs.new_ones((inp_length, target_length*2+1)) * neginf 
            alpha[0,0] = lprob[0, BLANK]

            dadi[0,0,0,BLANK] = 1

            if target_length > 0:
                alpha[0,1] = lprob[0, targets_prime[1]]
                dadi[0,1,0, targets_prime[1]] = 1

            for t in range(1, inp_length):
                for s in range(2 * target_length +1):
                    a1 = alpha[t-1,s]
                    a2 = alpha[t-1,s-1] if s > 0 else neginf
                    a3 = alpha[t-1,s-2] if s > 1 and targets_prime[s-2]!= targets_prime[s] else neginf

                    amax = max(a1,a2,a3)
                    amax = 0 if amax == neginf else amax
                    
                    alpha[t,s] = torch.log( torch.exp(a1-amax) + torch.exp(a2-amax) + torch.exp(a3-amax)) + \
                        amax + lprob[t, targets_prime[s]]  # (*)

                    #alpha[t,s] = func( lprob[k,l]) where k <= t. l can be any.
                    dadi[t,s,t,targets_prime[s]] = torch.exp(a1) + torch.exp(a2) + torch.exp(a3)

                    for k in reversed(range(t-1)):
                        for l in reversed(2 * target_length + 1):
                            #dadi[t,s,k,l] =  

                            # dadi[t,s,k,targets_prime[l]] = dadi[t-1,s-1,k,targets_prime[l]] * some lprob[..] +\
                            # dadi[t-1,s-2,k,targets_prime[l]] * some lprob[..]
                            pass


            #------------- construct betas & dadi
            beta          = log_probs.new_ones((inp_length, target_length * 2 + 1)) * neginf
            beta [-1, -1] = lprob[-1, BLANK]

            if target_length > 0: 
                beta[-1, -2] = lprob[-1,targets_prime[-2] ] # tricky

            for t in reversed(range(0, inp_length-1)):
                for s in reversed(range(0, 2*target_length+1)):
                    b1 = beta[t+1,s]
                    b2 = beta[t+1,s+1] if s < 2 * target_length else neginf
                    b3 = beta[t+1,s+2] if s < 2 * target_length -1 and targets_prime[s] != targets_prime[s+2] else  neginf

                    beta[t,s] = torch.log(torch.exp(b1)+torch.exp(b2)+torch.exp(b3)) + lprob[t,targets_prime[s]]

            #------------- aggregate gradients
        grad = None

        return grad_output * grad, None, None, None, None, None, None



class CTCLossFn(torch.autograd.Function):
    def forward(ctx, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, BLANK=0):
        # inp in log space
        input_lengths = torch.as_tensor(input_lengths, dtype=torch.int)
        target_lengths= torch.as_tensor(target_lengths, dtype=torch.int)
        cum_tgt_len   = target_lengths.cumsum(0)
        losses        = []
        alpha_global  = []

        for i in range(log_probs.size(1)):
            inp_length          = input_lengths[i].item()
            target_length       = target_lengths[i].item()
            cum_target_length   = cum_tgt_len[i].item()

            targets_prime = targets.new_full((2 * target_length + 1,), BLANK)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, :target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]

            lprob = log_probs[:inp_length, i]
            alpha   = log_probs.new_ones((inp_length, target_length*2+1)) * neginf 
            alpha[0,0] = lprob[0, BLANK]
            if target_length > 0:
                alpha[0,1] = lprob[0, targets_prime[1]]

            for t in range(1, inp_length):
                for s in range(2 * target_length +1):
                    a1 = alpha[t-1,s]
                    a2 = alpha[t-1,s-1] if s > 0 else neginf
                    a3 = alpha[t-1,s-2] if s > 1 and targets_prime[s-2]!= targets_prime[s] else neginf

                    amax = max(a1,a2,a3)
                    amax = 0 if amax == neginf else amax
                    
                    alpha[t,s] = torch.log( torch.exp(a1-amax) + torch.exp(a2-amax) + torch.exp(a3-amax)) + \
                        amax + lprob[t, targets_prime[s]]

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
            print( (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean() )
        elif reduction == 'sum':
            print( output.sum() )

        output_mean = (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()

        ctx.save_for_backward(log_probs,input_lengths, target_lengths, targets,torch.stack(alpha_global,dim=0), torch.tensor(losses))
        return output_mean       

    '''
    @staticmethod
    def forward_old(ctx, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank=0):
        # out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        batch_size = targets.size(0)
        max_target_length = targets.size(1)
        max_input_length = log_probs.size(0)
        targets_prime = targets.new_full((batch_size, 2 * max_target_length + 1,), blank)
        targets_prime[:, 1::2] = targets[:, :max_target_length]
        probs = log_probs.exp()
        
        # Initialization
        alpha = log_probs.new_zeros((batch_size, max_target_length * 2 + 1, ))
        alpha[:, 0] = probs[0, :, blank]
        alpha[:, 1] = probs[0].gather(1, targets_prime[:, 1].unsqueeze(-1)).squeeze(-1)
        mask_third = targets_prime[:, :-2] != targets_prime[:, 2:]
        for t in range(1, max_input_length):
            alpha_next = alpha.clone()
            alpha_next[:, 1:] += alpha[:, :-1]
            alpha_next[:, 2:] += torch.where(mask_third, alpha[:, :-2], torch.zeros_like(alpha[:, :-2]))
            alpha = probs[t].gather(1, targets_prime) * alpha_next
        out = -alpha[:, -2:].sum(-1).log()
        out = (out / target_lengths).mean()
        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths, alpha)
        return out
    '''
    
    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths, alphas, losses = ctx.saved_tensors
        return grad_output * CTCLossGradFn.apply(
            log_probs, targets, input_lengths, target_lengths, alphas, losses), \
            None, None, None
    
def compute_grad_loss(inputs, model, loss_fn, fl_grad):
    """
    Args:
        inputs
        model
        loss_fn
        fl_grad

    """
    log_probs = torch.nn.functional.log_softmax(
        model(inputs).permute(1, 0, 2))
    loss = loss_fn(log_probs)
    weights = [w for w in model.parameters() if w.requires_grad]
    weights_grad = torch.autograd.grad(loss, weights, create_graph=True)
    weights_grad = torch.cat([wg.view(-1) for wg in weights_grad], -1)
    grad_loss = torch.norm(weights_grad - fl_grad)
    return grad_loss

class TestCustomGradFn:
    def test_grad(self,
                  dim=8,
                  batch_size=2,
                  vocab_size=5,
                  input_max_len=10,
                  target_max_len=5):
        inputs = torch.autograd.Variable(
            torch.randn(batch_size, input_max_len, dim),
            requires_grad=True)  # B x L x d
        input_lengths = torch.ones(batch_size, dtype=torch.int) * input_max_len
        targets = torch.randint(1, vocab_size, (batch_size, target_max_len))
        target_lengths = torch.ones(batch_size, dtype=torch.int) * target_max_len
        
        model = torch.nn.Sequential(
            torch.nn.Linear(dim, 10),
            torch.nn.Linear(10, vocab_size))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fl_grad = torch.zeros(num_params)

        def loss_fn(x):
            return CTCLossFn.apply(x, targets, input_lengths, target_lengths)

        # compute gradient with our custom implementation
        grad_loss = compute_grad_loss(inputs, model, loss_fn, fl_grad)
        grad_loss.backward(retain_graph=True)
        x_grad = inputs.grad

        # compute gradient with finite difference (numerical)
        inputs_1 = inputs
        inputs_2 = inputs + torch.randn_like(inputs) * 0.01
        grad_loss_1 = compute_grad_loss(inputs_1, model, loss_fn, fl_grad)
        grad_loss_2 = compute_grad_loss(inputs_2, model, loss_fn, fl_grad)
        x_grad_num = (grad_loss_1 - grad_loss_2) / (inputs_1 - inputs_2)
        print("snr", 10 * torch.log10(torch.mean(torch.abs(x_grad) / torch.norm(x_grad - x_grad_num))).detach().numpy())
        exit()

    def test_ctc(self,
                 batch_size=3,
                 vocab_size=5,
                 input_max_len=10,
                 target_max_len=5):
        inputs = torch.randn(input_max_len, batch_size, vocab_size)
        inputs = torch.nn.functional.log_softmax(inputs)
        input_lengths = torch.ones(batch_size, dtype=torch.int) * input_max_len
        targets = torch.randint(1, vocab_size, (batch_size, target_max_len))
        target_lengths = torch.ones(batch_size, dtype=torch.int) * target_max_len

        ctc_built_in = torch.nn.functional.ctc_loss(inputs, targets, input_lengths, target_lengths)
        ctc_custom = CTCLossFn.apply(inputs, targets, input_lengths, target_lengths)
        print(ctc_built_in, ctc_custom)
        assert ctc_built_in - ctc_custom < 1e-5
    
    def test_ctc_grad(self,
                 batch_size=3,
                 vocab_size=5,
                 input_max_len=10,
                 target_max_len=5):
        inputs = torch.randn(input_max_len, batch_size, vocab_size)
        inputs = torch.nn.functional.log_softmax(inputs)
        input_lengths = torch.ones(batch_size, dtype=torch.int) * input_max_len
        targets = torch.randint(1, vocab_size, (batch_size, target_max_len))
        target_lengths = torch.ones(batch_size, dtype=torch.int) * target_max_len


        # test 1st order
        probs_1 = torch.nn.Parameter(inputs.data, requires_grad=True)
        probs_2 = torch.nn.Parameter(inputs.data, requires_grad=True)

        ctc_lib = torch.nn.functional.ctc_loss(probs_1, targets, input_lengths, target_lengths)
        ctc_imp = CTCLossFn.apply(probs_2,  targets, input_lengths, target_lengths)

        ctc_lib.backward()

        grad_1 = probs_1.grad.data
        grad_2 = CTCLossGradFn.apply(CTCLossFn.saved_tensors) # how to get saved tensors??


        print('norm:', torch.norm(grad_1-grad_2))

        assert torch.allclose(grad_1,grad_2) # <=== Passed this test 
        # # torch.autograd.gradcheck(ctc_loss_custom, (probs_1, inp_len, tgt_len, targets)) # <=== DIDN'T PASS THIS TEST WHEN COMPARE TO FINITE DIFFERENCES


 
 
    # def test_gradcheck(self):
    #     fn = CTCLossFn.apply
    #     inputs = torch.autograd.Variable(torch.randn(1, 10, 8, dtype=torch.float64), requires_grad=True)
    #     torch.autograd.gradcheck(fn, (inputs, None, None, None))

