import os
import torch
from socket import gethostname
from torch.utils.data import Dataset, DataLoader
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import time
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
import argparse
from torch import optim
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class DummyPretrainDataset(Dataset):
    def __init__(self, data_x, data_y, num_features, time_steps=15, seq_len=96, pred_len=96, label_len=48, bs=5, device=torch.device('mps'), in_mem=False):
        self.num_features = num_features
        self.time_steps = time_steps
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.batch_size = bs
        self.prebatch_len = seq_len + bs - 1
        self.device = device
        self.in_mem = in_mem
        if in_mem:
            self.data_x = torch.Tensor(data_x).to(self.device)
            self.data_y = torch.Tensor(data_y).to(self.device)
        else:
            self.data_x = data_x
            self.data_y = data_y

    def __len__(self):
        return (self.time_steps - self.seq_len - pred_len + 1) // self.batch_size

    def __getitem__(self, idx):
        s_begin = self.batch_size * idx
        s_end = s_begin + self.prebatch_len
        r_begin = s_begin + self.seq_len - self.label_len - 1
        r_end = r_begin + self.pred_len + self.label_len + self.batch_size - 1
        # print(s_begin, s_end, r_begin, r_end)

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if not self.in_mem:
            return torch.Tensor(seq_x).to(self.device), torch.Tensor(seq_y).to(self.device)
        else:
            return seq_x, seq_y


class SlidingWindowView:
    def __init__(self, window_size, stride, pred_len, label_len):
        self.window_size = window_size
        self.stride = stride
        self.pred_len = pred_len
        self.label_len = label_len

    def slide_collate_fn(self, batch):
        return (batch[0].unfold(0, self.window_size, self.stride).transpose(1, 2),
                batch[1].unfold(0, self.pred_len+self.label_len, self.stride).transpose(1, 2))


class DummyPretrainStackDataset(Dataset):
    def __init__(self, data_x, data_y, num_features, time_steps=15, seq_len=6, pred_len=8, label_len=8, bs=5):
        self.num_features = num_features
        self.time_steps = time_steps
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return self.time_steps - self.seq_len + 1 - self.pred_len  # // self.batch_size

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len + self.label_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return torch.Tensor(seq_x), torch.Tensor(seq_y)


def get_args(parser):

    #benchmarking
    parser.add_argument('--mode', type=str, default='regular', help='Profiling system: "regular", "optimize" ')
    parser.add_argument('--device', type=str, default='mps', help='Device')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--num_features', type=int, default=10, help='number of input features')
    parser.add_argument('--time_steps', type=int, default=3000, help='number of input time steps')

    # # data loader
    # parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=500, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=40, help='start token length')
    parser.add_argument('--pred_len', type=int, default=60, help='prediction sequence length')

    # DLinear
    # parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    # parser.add_argument('--enc_in', type=int, default=num_features,
    #                     help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    # parser.add_argument('--dec_in', type=int, default=num_features, help='decoder input size')
    # parser.add_argument('--c_out', type=int, default=num_features, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--n_epoch', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    args.enc_in = args.num_features
    args.dec_in = args.num_features
    args.c_out = args.num_features

    import random
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    return args


def train_regular_one_epoch(args, dl, model, optimizer, loss_fn):
    running_loss = 0.

    start = time.time()
    for n_batch, data in enumerate(dl):
        # Every data instance is an input + label pair
        inputs, labels = data
        # if 'optimize' not in args.mode:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        dec_inp = torch.zeros_like(labels[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([labels[:, :args.label_len, :], dec_inp], dim=1).float() #.to(device)

        if 'Linear' in m or 'TST' in m:
            outputs = model(inputs)
        else:
            # outputs = model(inputs)
            outputs = model(inputs, inputs[:, :, :4], dec_inp, labels[:, :, :4])

        outputs = outputs[:, -args.pred_len:, :]
        labels = labels[:, -args.pred_len:, :]

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
    stop = time.time()
    # avg_loss = running_loss / len(dl)
    # if args.mode == 'regular':
    #     print('Stacked || ', m, ': ', avg_loss, ' | duration: ', stop - start)
    # else:
    #     print('Optimize || ', m, ': ', avg_loss, ' | duration: ', stop - start)
    return stop - start


if __name__ == '__main__':
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)


    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    args = get_args(parser)
    print('Args:', args)

    if args.device == 'mps':
        device = torch.device(args.device)
        torch.mps.empty_cache()
    else:
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        device = torch.device(args.device)

    time_steps = args.time_steps
    num_features = args.num_features
    pred_len = 60
    label_len = 40
    data_x = np.random.uniform(low=1, high=100, size=(time_steps, num_features))
    data_y = data_x
    print(data_x.shape, data_y.shape)

    if args.mode == 'optimize':
        dset = DummyPretrainDataset(data_x, data_y, num_features, time_steps, seq_len=args.seq_len, pred_len=pred_len,
                                    label_len=label_len, bs=args.batch_size, device=device)
        sw = SlidingWindowView(args.seq_len, 1, pred_len, label_len)
        sampler = DistributedSampler(dset, rank=rank, shuffle=False, drop_last=True)
        dl = DataLoader(dset, batch_size=None, collate_fn=sw.slide_collate_fn,
                        pin_memory=False, num_workers=0, sampler=sampler)
    elif args.mode == 'optimize_in_mem':
        dset = DummyPretrainDataset(data_x, data_y, num_features, time_steps, seq_len=args.seq_len, pred_len=pred_len,
                                    label_len=label_len, bs=args.batch_size, device=device, in_mem=True)
        sw = SlidingWindowView(args.seq_len, 1, pred_len, label_len)
        sampler = DistributedSampler(dset, rank=rank, shuffle=False, drop_last=True)
        dl = DataLoader(dset, batch_size=None, collate_fn=sw.slide_collate_fn, pin_memory=False, num_workers=0, sampler=sampler)
    else:
        sset = DummyPretrainStackDataset(data_x, data_y, num_features, time_steps, seq_len=args.seq_len, pred_len=pred_len,
                                         label_len=label_len)
        sampler = DistributedSampler(sset, rank=rank, shuffle=False, drop_last=True)
        dl = DataLoader(sset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0, sampler=sampler)

    models = {'PatchTST': PatchTST, 'Informer': Informer, 'Autoformer': Autoformer, 'DLinear': DLinear, 'Transformer': Transformer}
    print('# Batches: ', len(dl))

    num_epoch = 0
    warmup_epoch = 5
    total_runtime = 0
    for m in models.keys():

        args.model = models[m]
        model = args.model.Model(args).to(device)
        model = DDP(model, device_ids=[local_rank])
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(m, ': ', params, ' params')
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.MSELoss()

        for i in range(warmup_epoch):
            train_regular_one_epoch(args, dl, model, optimizer, loss_fn)

        start = time.time()
        for i in range(args.n_epoch - warmup_epoch):
            runtime = train_regular_one_epoch(args, dl, model, optimizer, loss_fn)
            if args.mode == 'regular':
                print('Stacked ||  duration: ', runtime)
            elif args.mode == 'optimize_in_mem':
                print('Optimize in memory ||  duration: ', runtime)
            else:
                print('Optimize || duration: ', runtime)
        end = time.time()
        total_runtime = end - start
        if rank == 0: print('Total runtime: ', total_runtime, " | Avg. runtime: ", total_runtime / (args.n_epoch - warmup_epoch))
    dist.destroy_process_group()
