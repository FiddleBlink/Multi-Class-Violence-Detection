from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test import test
import option
import time


if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/wsanodetV3.pkl').items()})
    gt = np.load(args.gt)
    st = time.time()
    pr_auc, f1, precision1, recall1, accuracy = test(test_loader, model, device, gt)
    print('Time:{}'.format(time.time()-st))
    print('offline pr_auc:{0:.4}\n'.format(pr_auc))
    print('offline f1:{0:.4}\n'.format(f1))
    print(f'offline precision: {precision1}, offline Recall: {recall1}\n')
    print(f'Sklearn Accuracy Score: {accuracy}\n')


