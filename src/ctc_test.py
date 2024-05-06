import torch
from .ctc import CustomCTCLoss


class TestCTC:
    def test_ctc_val(self):
        device    = 'cpu'
        T  = 100
        C_SIZE = 5

        tgt_len = [5,5,5,5]
        inp_len = [T,T,T,T]
        BS = len(tgt_len)

        targets = torch.randint(BS, C_SIZE , (sum(tgt_len),), dtype=torch.int)
        log_probs = torch.nn.Parameter((torch.randn(T, BS, C_SIZE, dtype=torch.float, device=device).softmax(2))).log()

        ctc_loss = torch.nn.CTCLoss()
        ctc_loss_custom = CustomCTCLoss.apply
        out_built_in = ctc_loss(log_probs, targets, inp_len, tgt_len) # in log space
        out_custom = ctc_loss_custom(log_probs, inp_len, tgt_len, targets)  # in soft max space

        print(out_built_in, out_custom)
        assert torch.norm(out_built_in - out_custom) < 1e-4

    # def test_ctc_grad(self):
    #     ctc_loss_custom = CustomCTCLoss.apply
    #     device    = 'cpu'
    #     BS = 1
    #     T  = 20
    #     C_SIZE = 15

    #     tgt_len = [30]
    #     inp_len = [50]
    #     targets = torch.randint(1, C_SIZE , (sum(tgt_len),), dtype=torch.int)
    #     log_probs = torch.nn.Parameter((torch.randn(T, BS, C_SIZE, dtype=torch.double, device=device).softmax(2))).log()
    #     torch.autograd.gradcheck(ctc_loss_custom, (log_probs, inp_len, tgt_len, targets))
    
    def test_ctc_grad(self):
        ctc_loss_custom = CustomCTCLoss.apply
        device    = 'cpu'
        T  = 100
        C_SIZE = 20

        tgt_len = [80,80,80]
        inp_len = [T,T,T]
        BS = len(tgt_len)


        targets = torch.randint(1, C_SIZE , (sum(tgt_len),), dtype=torch.int)

        probs_1 = torch.nn.Parameter(torch.randn(T, BS, C_SIZE, dtype=torch.double, device=device).softmax(2).log(), requires_grad=True)
        probs_2 = torch.nn.Parameter(probs_1.data, requires_grad=True)

        ctc_lib = torch.nn.functional.ctc_loss(probs_1, targets, inp_len, tgt_len)
        ctc_imp = ctc_loss_custom(probs_2, inp_len, tgt_len, targets)

        assert torch.allclose(ctc_lib ,ctc_imp )


        ctc_lib.backward()
        grad_1 = probs_1.grad.data

        ctc_imp.backward()
        grad_2 = probs_2.grad.data


        print('norm:', torch.norm(grad_1-grad_2))

        assert torch.allclose(grad_1,grad_2) # <=== Passed this test 
        # torch.autograd.gradcheck(ctc_loss_custom, (probs_1, inp_len, tgt_len, targets)) 
        # <=== DIDN'T PASS THIS TEST WHEN COMPARE TO FINITE DIFFERENCES

