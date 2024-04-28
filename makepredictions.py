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
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/Binary Inverse Adam/wsanodet_Adam_Binary_Inverse_50.pkl').items()})
    
    model.eval()
    with torch.no_grad():
        pred = torch.zeros(0).to(device)

        for i, input in enumerate(test_loader):
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            
            probs = torch.softmax(logits, 2)
            probs = torch.mean(probs, dim=0)  # Fix: Replace 0 with dim=0
            pred = torch.argmax(probs, 1).float()

            probabilities = list(pred.cpu().detach().numpy())
            probabilities = [round(num, 3) for num in probabilities]

            time_stramp = {
                "fighting":[],
                "shooting":[],
                "explosion":[],
                "riot":[],
                "abuse":[],
                "accident":[]
            }

            occurence = 0

            activity = set(probabilities)

            for i in activity:
                for j in range(len(probabilities)-1):
                    if(i == probabilities[j] and occurence == 0):
                        start_frame = j
                        start_timestamp = (start_frame * 16)/ 24
                    if(i == probabilities[j]):
                        occurence = occurence + 1
                    if( (i == probabilities[j] and i != probabilities[j+1]) or ( i == probabilities[j] and j == len(probabilities)-2 )):
                        occurence = 0
                        end_frame = j
                        end_timestamp = (end_frame * 16)/ 24
                        if(end_timestamp - start_timestamp < 3):
                            continue
                        if(i==1):
                            time_stramp["fighting"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                        elif(i==2):
                            time_stramp["shooting"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                        elif(i==3):
                            time_stramp["explosion"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                        elif(i==4):
                            time_stramp["riot"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                        elif(i==5):
                            time_stramp["abuse"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})
                        elif(i==6):
                            time_stramp["accident"].append({"start_time":round(start_timestamp, 2),"end_time":round(end_timestamp, 2)})

            print(time_stramp)
            