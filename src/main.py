# this is the main script to reconstruct data point for DS2 arch
import argparse
from ast import arg
import logging
import numpy as np
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose

# Configure basic logging (will be updated with file handler in main)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Add module paths
sys.path.insert(0, os.path.abspath('../modules/deepspeech/src'))
sys.path.insert(0, os.path.abspath('../src/'))

# Local imports
from ctc.ctc_loss_imp import *
from data.librisubset import *
from loss.loss import *
from models.ds2 import DeepSpeech2
from utils.plot import *
from utils.util import *

def get_device_net(args, use_relu):
    device = 'cuda:0'
    net = DeepSpeech2(winlen=0.032, winstep=0.02).to(device)
    return device, net

def zero_order_optimization_loop(inputs, x_param, output_sizes, target_size,
                                 net,
                                    dldw_targets , params_to_match, targets, prefix, args):
    device = inputs.device
    net.eval()

    loss_func = lambda x,y :batched_ctc_v2(x, y, output_sizes, target_size)
    

    i = 0 
    stop_condition = False

    def get_meta_loss(x_param):
        out = net(x_param)
        out = out.log_softmax(-1)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets, params_to_match, loss_func, args)
        return mloss, dldws
    
    tolerance = 10
    step_size = args.zero_order_lr

    loss_history = []
    loss_gm_history = []
    loss_reg_history = []

    while i < args.zero_max_iterations and not stop_condition:
        # random 16 directions in the space of x_param
        directions = torch.randn(8, *x_param.shape).to(device)
        # normalize direction so that they have unit length
        shape = directions.shape
        directions = directions.reshape(8,-1)
        directions = torch.functional.F.normalize(directions, dim=1)
        directions = directions.reshape(shape)
        # create a list to store the loss for each direction
        losses = []
        current_loss, _ = get_meta_loss(x_param)
        # logging.info('Current loss: {}'.format(current_loss.item()))

        for d in directions:
            x_param_new = x_param + step_size * d
            mloss, _ = get_meta_loss(x_param_new)
            losses.append(mloss.item())

        # find the best direction by averaging direction that reduce loss
        # best_direction = directions[torch.tensor(losses) < current_loss.item()]
        # find the best direction by the smallest loss, argmin
        best_direction = directions[np.argmin(losses)]

        if  np.min(losses) < current_loss.item():
            # best_direction = best_direction.mean(dim=0)
            x_param = x_param + step_size * best_direction
            tolerance = 10
            step_size = args.zero_order_lr
        else:  
            logging.info('No direction found, reducing step size, tolerance: {}, {}'.format(step_size,tolerance))
            tolerance -=1
            step_size *= 0.5
            if tolerance < 0:
                stop_condition = True

        mae = torch.mean(torch.abs(x_param - inputs))
        logging.info('iter {}  loss: {}, step size: {}, mae: {}'.format(i, np.min(losses), step_size, mae.item()))


        loss_history.append(current_loss.item())    
        loss_gm_history.append(0)
        loss_reg_history.append(0)
        if i % 20 == 0:
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i, prefix=prefix, args=args)
            pass
 
        i += 1

    return x_param

def first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes,
                                  optimizer, scheduler, net,
                                  dldw_targets , params_to_match, targets,prefix,  args):

    net.train()
    loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')

    i=0
    loss_history = []
    loss_gm_history = []
    loss_reg_history = []
    stop_condition = False
    while i < args.max_iterations and not stop_condition:
        # x_param_full= torch.concat([x_param, x_pad], dim=2)
        out = net(x_param) # 1 176 29
        out = out.log_softmax(-1)
        # mloss, dldw_f = meta_loss(output, targets, output_sizes, target_sizes, dldw_target,  weight_param)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets,  params_to_match, loss_func, args)
        gm_weight_distance = grad_distance(dldws[0], dldw_targets[0], args)
        gm_bias_distance   =  0.0 #grad_distance(dldws[1], dldw_targets[1], args)

        # regloss = tv_norm(x_param)
        if args.regularization == 'L2':
            regloss = torch.norm(x_param, p=2)
        elif args.regularization == 'L1':
            pass
        elif args.regularization == 'TV':
            # need to make x_param from [n_frame, batch size, n_features] to [batch size, 1, n_features, n_frame]
            regloss = tv_norm(x_param.permute(1,0,2).unsqueeze(1))
        else:
            regloss = torch.tensor(0.0)
       
        loss = (1-args.reg_weight)* mloss + args.reg_weight * regloss



        optimizer.zero_grad()
        loss.backward()
        grad = x_param.grad.data

        # torch.nn.utils.clip_grad_norm_(x_param, 1.0)
        optimizer.step()
        scheduler.step()


        mae = torch.mean(torch.abs(x_param - inputs))

        loss_history.append(loss.item())
        loss_gm_history.append(mloss.item() )
        loss_reg_history.append(regloss.item() )

        if i % 10 == 0:
            logging.info('Iter, Loss (A-G-Gw-Gb-R), Gradient Norm, Learning Rate, MAE: {:4d}, {:.8f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'\
                        .format(i, loss.item(), mloss.item(),  gm_weight_distance.item(), gm_bias_distance, regloss.item()
            , grad.norm().item(), optimizer.param_groups[0]["lr"], mae.item()))
            # scheduler.step(mloss.item())

        if i % 100 == 0:
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i,prefix=prefix, args=args)
            pass
            
        
        i+=1
        # stet stop condition true if loss not decrease in last 100 iteration
        if i>100 and loss_history[-1] > min(loss_history[-100:]):
            stop_condition = True
        else:
            stop_condition
    return x_param


# ---------------------------------------------------------------------------- #
#                      create a optimization loop function                     #
# ---------------------------------------------------------------------------- #
def optimization_loop(inputs, x_param, output_sizes, target_sizes,
                       optimizer, scheduler, net, 
                       dldw_targets , params_to_match, targets,prefix='',  args=None):

    device = inputs.device

    # loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')

    if not os.path.exists(os.path.join(args.exp_path, prefix+'_x_param_first_order.pt')) or not args.resume_from_first_order:
        logging.info('Running first order optimization loop')
        x_param = first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, optimizer, scheduler, net, dldw_targets, params_to_match, targets,prefix+'_firstorder', args) 
        torch.save(x_param.detach().cpu(), os.path.join(args.exp_path, prefix+'_x_param_first_order.pt'))
        logging.info('x_param_first_order.pt saved')
    else:
        x_param = torch.load(os.path.join(args.exp_path, prefix+'_x_param_first_order.pt')).to(device)
        logging.info('x_param_first_order.pt loaded')

    if args.use_zero_order_optimization:
        logging.info('Running zero order optimization loop')
        x_param = zero_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, net, dldw_targets, params_to_match, targets,prefix+'_zeroorder',args)


    return x_param

# ---------------------------------------------------------------------------- #
#                 Reconstruct all datapoint in a torch dataset                 #
# ---------------------------------------------------------------------------- #
def reconstruct_dataset(network, device, dataloader, args):
    torch.manual_seed(0)

    # loop through item in the dataloader
    for (i, batch) in enumerate(dataloader):


        # batch = A tuple of `((batch_x, batch_out_lens), batch_y)` where:
        logging.info('#'*20)
        logging.info('Processing batch {}/{}'.format(i, len(dataloader)))

        inputs ,input_sizes = batch[0]
        logging.info('inputs mean and std: {}, {}'.format(inputs.mean(), inputs.std()))
        # input_sizes is tensor list of inputs.shape[1] elements with value inputs.shape[0]

        targets = batch[1]
        text = ''.join(network.ALPHABET.get_symbols(targets[0].tolist()))
        logging.info('TEXT: {}'.format(text))
        target_sizes = torch.Tensor([len(t) for t in targets]).int()

        #target is list of tensor with different length, pad it to the same length in a tensor
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)


        # transfer the data to the GPU
        inputs = inputs.to(device)
        targets = targets.long().to(device)

        input_sizes = input_sizes.long().to(device)
        target_sizes = target_sizes.long().to(device)


        out = network(inputs)


        # params_to_match = [network.network.fc.module[1].weight, network.network.out.module[0].bias]
        params_to_match = [network.network.fc.module[1].weight] # deep speech 2 doesnt use bias in last FC
        output_sizes = (torch.ones(out.shape[1]) * out.shape[0]).int()
        out =  out.log_softmax(-1)

        loss_func = lambda x,y : batched_ctc_v2(x, y, output_sizes, target_sizes)

        loss = loss_func(out, targets)
        # loss_func_lib   = torch.nn.CTCLoss()
        # loss_lib = loss_func_lib(out.cpu(), targets.cpu(), output_sizes.cpu(), target_sizes.cpu())

        logging.debug('loss: {}'.format(loss.item()))
        dldw_targets = torch.autograd.grad(loss, params_to_match)

        ## zero out small values keep 10% largest dldw_target
        # logging.info('zero out small values keep 10% largest dldw_target')
        # dldw_target = dldw_target * (dldw_target.abs() > dldw_target.abs().topk(int(0.1*dldw_target.numel()))[0][-1])
        for ip, p in enumerate(params_to_match):
            p.requires_grad = True
            logging.debug('matching {}. params with shape {} and norm {} first ten {}'.format(ip, p.shape, p.norm(), p.flatten()[:10]))
            logging.debug('                    gradient norm {}'.format(dldw_targets[ip].norm()))


        x_init =  init_a_point(inputs, args)
        x_param = torch.nn.Parameter(x_init.to(device),requires_grad=True)


        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam([x_param], lr=args.learning_rate)
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD([x_param], lr=args.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        # reduce lr at epoch 250, 500, 750 half
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(250,2000,250)), gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.5,patience=50)


        # suggest an experiment name base on datapoint index, optimizer name,  learning rate, regularizer, regularizer weight
        logging.info('Experiment Name: {}'.format(os.path.basename(args.exp_path)))

        # timing the optimization loop
        start_time = time.time()
        x_param = optimization_loop(inputs, x_param, output_sizes, target_sizes,
                        optimizer, scheduler, network,
                        dldw_targets = dldw_targets, params_to_match =  params_to_match, targets = targets, prefix=f'sampleidx_{i}', args=args)
        end_time = time.time()
                        
        save_path = os.path.join(args.exp_path, f'sampleidx_{i}_' + 'x_param_last.pt'.format(i))
        # save x_param, optimization time, inputs, targets
        torch.save({ 'x_param': x_param.detach().cpu(),
                    'time': end_time - start_time,
                    'inputs': inputs.detach().cpu(), 
                    'targets': targets.detach().cpu(),
                    'transcript': text
                    }, save_path)

def main(args):
    """
    Main function for reconstructing data points with specified hyperparameters.
   
    Parameters:
    - index: Index of the data point to reconstruct.
    - lr: Learning rate for optimization.
    - reg: Type of regularization ('L1', 'L2', 'TV').
    - reg_weight: Weight of the regularization term.
    - iterations: Number of iterations for the optimization.
    """
    
    #check if exp_path exists or create
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    if not os.path.exists(os.path.join(args.exp_path, 'figures')):
        os.makedirs(os.path.join(args.exp_path, 'figures'))
    
    # Set up file logging now that exp_path exists
    log_file = os.path.join(args.exp_path, 'experiment.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Get root logger and add file handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logging.info('='*80)
    logging.info('Starting new experiment run')
    logging.info('='*80)
    
    # Log all arguments
    logging.info('Experiment configuration:')
    for key, value in args.__dict__.items():
        logging.info('  {}: {}'.format(key, value))
    logging.info('-'*80)
    
    # Change all print statements to logging statements
    logging.info('Optimizer: {}'.format(args.optimizer))
    logging.info('Learning rate: {}'.format(args.learning_rate))
    logging.info('Regularization: {}'.format(args.regularization))
    logging.info('Regularization weight: {}'.format(args.reg_weight))
    logging.info('Number of iterations: {}'.format(args.max_iterations))
    logging.info('exp_path: {}'.format(args.exp_path))
    logging.info('Log file: {}'.format(log_file))

    # net & devices
    device, net = get_device_net(args, use_relu=False)
    logging.info('Device: {}'.format(device))
    logging.info('Network: {}'.format((net.__class__.__name__)))

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path)['network']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = 'network.' + k  # Add 'network.' prefix
            new_state_dict[new_key] = v
        net.load_state_dict(new_state_dict)
        logging.info('Checkpoint loaded from {}'.format(args.checkpoint_path))
    else:
        # loging random init weight
        logging.info('Random init weight')

    # dataset & dataloader
    dataset, loader = get_dataset_libri_sampled_folder_subset(net, args)

    # run reconstruct
    reconstruct_dataset(net, device, loader, args)
    
    # Log completion
    logging.info('='*80)
    logging.info('Experiment completed successfully')
    logging.info('='*80)


def pretty_print_config(args):
    for key, value in args.__dict__.items():
        print("{:30} {}".format(key, value))
    
def parse_args():
    '''
     example run to reconstruct data point index 0 with lr 0.001, distance cosine, reg weight 0, min duration 1sec max 2 sec.  otpimizer Adam , none regularizer, 2000, 100% top gradient
     python src/main.py --batch_start_idx 0 --batch_end_idx 1 --learning_rate 0.001 \
         --distance_metric cosine --reg_weight 0.0 --min_duration_ms 1000 --max_duration_ms 2000 --optimizer Adam --regularization None --max_iterations 2000 --top_grad_percentage 1.0
         
    '''
    
    parser = argparse.ArgumentParser(description="Reconstruct a data point with specified parameters.")

    # env params
    parser.add_argument("--dataset_path"         , type=str  , default='./datasets/librispeech_sampled_600_file_0s_4s', help="Dataset path, symlinked from /scratch/f006pq6/datasets")
    parser.add_argument("--checkpoint_path"      , type=str  , default=None      , help="Path to checkpoint to resume from")
 
    # data params
    parser.add_argument("--min_duration_ms"      , type=int  , default=0         , help="Minimum duration of batch in milliseconds")
    parser.add_argument("--max_duration_ms"      , type=int  , default=1000      , help="Maximum duration of batch in milliseconds")
    parser.add_argument("--batch_start_idx"      , type=int  , required=True , help="Starting index of the batch"                       , default=0)
    parser.add_argument("--batch_end_idx"        , type=int  , required=True , help="Ending index of the batch"                         , default=1)
    parser.add_argument("--batch_size"           , type=int  , default=1         , help="Batch size")
 
    # model params
    parser.add_argument("--dropout_prob"         , type=float, default=0.0       , help="Dropout probability")
    parser.add_argument("--context_frames"       , type=int  , default=6         , help="Number of context frames")

    # optimization params
    parser.add_argument("--resume_from_first_order", action='store_true', help="Whether to resume from first order optimization checkpoint", default=False)
    parser.add_argument("--num_seeds"            , type=int  , default=10        , help="Number of random seeds to try")
    parser.add_argument("--optimizer"            , type=str  , default='Adam'    , help="Optimizer to use for optimization")
    parser.add_argument("--learning_rate"        , type=float, default=0.01      , help="Learning rate for first-order optimization")
    parser.add_argument("--regularization"       , type=str  , default='None'    , choices=["L1", "L2", "TV", "None"], help="Type of regularization")
    parser.add_argument("--reg_weight"           , type=float, default=0.0       , help="Weight of the regularization term")
    parser.add_argument("--max_iterations"       , type=int  , default=2000      , help="Maximum iterations for first-order optimization")
    parser.add_argument("--top_grad_percentage"  , type=float, default=1.0       , help="Percentage of top gradients to keep")
    parser.add_argument("--distance_metric"      , type=str  , default='cosine'  , help="Distance metric for gradient matching")
    parser.add_argument("--distance_metric_weight", type=float, default=0.5      , help="Weight for combining distance metrics (used in cosine+l2)")
    parser.add_argument("--initialization_method", type=str  , default='uniform' , help="Method for initializing input reconstruction")


    parser.add_argument("--use_zero_order_optimization", action='store_true', help="Whether to use zero-order optimization after first-order optimization", default=False)
    parser.add_argument("--zero_order_lr"        , type=float, default=100       , help="Learning rate for zero-order optimization")
    parser.add_argument("--zero_max_iterations"  , type=int  , default=200       , help="Maximum iterations for zero-order optimization")

    args = parser.parse_args()

    assert args.batch_size == 1, "Batch size must be 1"
    return args

def set_up_dir(args):
    if args.min_duration_ms == 0 and args.max_duration_ms == 1000:
        exp_path='logging/0s-1s/'
    elif args.min_duration_ms == 1000 and args.max_duration_ms == 2000:
        exp_path='logging/1s-2s/'
    elif args.min_duration_ms == 2000 and args.max_duration_ms == 3000:
        exp_path='logging/2s-3s/'
    elif args.min_duration_ms == 3000 and args.max_duration_ms == 4000:
        exp_path='logging/3s-4s/'
    
    cpt_name = os.path.basename(args.checkpoint_path) if args.checkpoint_path is not None else 'None'
    exp_name = f"DS2_batchstart_{args.batch_start_idx}_batch_end_{args.batch_end_idx}_init_{args.initialization_method}_opt_{args.optimizer}_lr_{args.learning_rate}_reg_{args.regularization}_regw_{args.reg_weight}_top-grad-perc_{args.top_grad_percentage}_cpt_{cpt_name}"
    args.exp_path=os.path.join(exp_path, exp_name)
    return args

if __name__ == "__main__":
    # args & dir
    args = parse_args()
    args = set_up_dir(args)
    pretty_print_config(args)
    # main
    main(args)