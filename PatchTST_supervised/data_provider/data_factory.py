from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, device=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        batch_size=args.batch_size,
        optimize=args.optimize,
        device=device

    )
    print(flag, len(data_set))

    if args.optimize == 1:
        sw = SlidingWindowView(args.seq_len, 1, args.pred_len, args.label_len)
        dl = DataLoader(data_set, batch_size=None, collate_fn=sw.slide_collate_fn, pin_memory=False)
        return data_set, dl
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

class SlidingWindowView:
    def __init__(self, window_size, stride, pred_len, label_len):
        self.window_size = window_size
        self.stride = stride
        self.pred_len = pred_len
        self.label_len = label_len

    def slide_collate_fn(self, batch):
        return (batch[0].unfold(0, self.window_size, self.stride).transpose(1, 2),
                batch[1].unfold(0, self.pred_len+self.label_len, self.stride).transpose(1, 2),
                batch[2].unfold(0, self.window_size, self.stride).transpose(1, 2),
                batch[3].unfold(0, self.pred_len + self.label_len, self.stride).transpose(1, 2)
                )
