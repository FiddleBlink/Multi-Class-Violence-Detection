from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
import os
from model import Model
from dataset import Dataset
from train import train
from test import test
import option


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # setup_seed(2333)
    args = option.parser.parse_args()
    device = torch.device("cuda")
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)


    device = torch.device('cuda:{}'.format(args.gpus) if args.gpus != '-1' else 'cpu')
    model = Model(args).to(device)

    # for name, value in model.named_parameters():
    #     print(name)
    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    optimizer = optim.Adam([{'params': base_param},
                            {'params': model.approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
                            ],
                            lr=args.lr, weight_decay=0.000)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCELoss()

    is_topk = True
    gt = np.load(args.gt)

    latestepoch = 0
    # if os.path.exists('./ckpt/'+args.model_name+'{}.pkl'.format(latestepoch)):
    #     model.load_state_dict(torch.load('./ckpt/' + args.model_name + '{}.pkl'.format(latestepoch)))

    # pr_auc, pr_auc_online, f1, precision1, recall1, accuracy = test(test_loader, model, device, gt)
    # print('Random initalization: offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    
    train_losses = []
    accuracy_arr = []
    f1_arr = []
    precision_arr = []
    recall_arr = []
    roc_auc_arr = []
    
    for epoch in range(args.max_epoch - latestepoch):
        print(f'[INFO] EPOCH No.{epoch + 1 + latestepoch} under Processing=====\n\n')
        
        scheduler.step()
        st = time.time()
        loss = train(train_loader, model, optimizer, criterion, criterion2, device, is_topk)
        train_losses.append(loss)
        
        if epoch % 2 == 0 and not epoch == 0:
            torch.save(model.state_dict(), './ckpt/'+args.model_name+'{}.pkl'.format(epoch))

        roc_auc, f1, precision1, recall1, accuracy = test(test_loader, model, device, gt)
        # print('Epoch {0}/{1}: offline roc_auc:{2:.4}'.format(epoch, args.max_epoch, roc_auc))
        accuracy_arr.append(accuracy)
        f1_arr.append(f1)
        precision_arr.append(precision1)
        recall_arr.append(recall1)
        roc_auc_arr.append(roc_auc)

    np.save('./ckpt/train_losses.npy'.format(epoch), np.array(train_losses))
    np.save('./ckpt/roc_auc.npy'.format(epoch), np.array(roc_auc_arr))
    np.save('./ckpt/f1.npy'.format(epoch), np.array(f1_arr))
    np.save('./ckpt/precision.npy'.format(epoch), np.array(precision_arr))
    np.save('./ckpt/recall.npy'.format(epoch), np.array(recall_arr))
    np.save('./ckpt/accuracy.npy'.format(epoch), np.array(accuracy_arr))

    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
