from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import numpy as np
import torch

def test(dataloader, model, device, gt):
    with torch.no_grad():
        model.eval()
        all_preds = []
        all_probs = np.zeros((0, 7))
        for i, input in enumerate(dataloader):
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            probs = torch.softmax(logits, 2)
            probs = torch.mean(probs, dim=0)  # Fix: Replace 0 with dim=0
            pred = torch.argmax(probs, 1).float()
            all_preds.extend(pred.cpu().numpy())
            all_probs = np.concatenate((all_probs, probs.view(-1, probs.size(-1)).cpu().numpy()))

        all_preds = np.array(all_preds)
        print(f'pred: {all_preds.shape}')
        print(f'pred: {all_preds[:100]}')


        # pred: 145649
        # GT: 2330384

        roc_auc = roc_auc_score(list(gt), np.repeat(all_probs, 16, axis=0), multi_class='ovr', average='weighted')
        f1 = f1_score(list(gt), np.repeat(all_preds, 16), average='weighted')
        prec = precision_score(list(gt), np.repeat(all_preds, 16), average='weighted')
        recal = recall_score(list(gt), np.repeat(all_preds, 16), average='weighted')
        acc = accuracy_score(list(gt), np.repeat(all_preds, 16))

        print(f'ROC AUC: {roc_auc}')
        print(f'F1: {f1}')
        print(f'Precision: {prec}')
        print(f'Recall: {recal}')
        print(f'Accuracy: {acc}')

        return roc_auc, f1, prec, recal, acc