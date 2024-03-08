from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch
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
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/wsanodet_mix2.pkl').items()})
    
    model.eval()
    with torch.no_grad():
        pred = torch.zeros(0).to(device)

        for i, input in enumerate(test_loader):
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            

            print(logits.shape)
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))
            probabilities = list(pred.cpu().detach().numpy())
            probabilities = [round(num, 3) for num in probabilities]
            print(probabilities)
            print(logits.shape)
            