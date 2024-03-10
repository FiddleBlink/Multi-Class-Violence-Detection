from sklearn.metrics import auc, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import numpy as np
import torch

def test(dataloader, model, device, gt):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            logits = torch.squeeze(logits)
            sig = torch.softmax(logits, 0)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))
            '''
            online detection
            '''
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sig2 = torch.mean(sig2, 0)

            sig2 = torch.unsqueeze(sig2, 1) ##for audio
            pred2 = torch.cat((pred2, sig2))

        print(f'pred: {pred.shape}')
        pred = list(pred.cpu().detach().numpy())
        pred2 = list(pred2.cpu().detach().numpy())


        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        pr_auc2 = auc(recall, precision)
        
        
        pred = [(p > 0.75).astype(int) for p in pred]
        f1 = f1_score(list(gt), np.repeat(pred, 16))

        precision1 = precision_score(list(gt), np.repeat(pred, 16))
        recall1 = recall_score(list(gt), np.repeat(pred, 16))

        matrix = confusion_matrix(list(gt), np.repeat(pred, 16))
        matrix = np.flip(matrix)

        accuracy = accuracy_score(list(gt), np.repeat(pred, 16))

        return pr_auc, pr_auc2, f1, precision1, recall1, accuracy



