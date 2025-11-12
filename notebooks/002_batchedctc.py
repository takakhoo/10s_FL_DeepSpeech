class CTCLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank=0):
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