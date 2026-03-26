from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
import os
import logging
from model import Model
from dataset import Dataset
from train import train
from test import test
import option

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f'Random seed set to {seed}')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = option.parser.parse_args()
    
    logging.info(f'=== XDVioDet Training Started ===')
    logging.info(f'Mode: {args.online_mode} | Weights: {args.weights} | Optimizer: {args.optimizer}')
    logging.info(f'LR: {args.lr} | Batch Size: {args.batch_size} | Max Epochs: {args.max_epoch}')
    
    # Setup seed if provided
    if args.seed is not None:
        setup_seed(args.seed)
    
    device = torch.device('cuda:{}'.format(args.gpus) if args.gpus != '-1' else 'cpu')
    logging.info(f'Device: {device}')

    # Setup class weights if using inverse weighting
    class_weights = None
    if args.weights == 'Inverse':
        logging.info('Computing inverse class weights...')
        train_data_all = DataLoader(Dataset(args, test_mode=False), batch_size=args.batch_size, shuffle=True)
        all_labels = []

        for i, (input, label) in enumerate(train_data_all):
            all_labels.append(label)
            if (i + 1) % 50 == 0:
                logging.info(f'Processed {i + 1} batches for weight calculation')

        all_labels = torch.cat(all_labels, dim=0)
        logging.info(f'Total labels loaded: {all_labels.shape}')

        torch_labels = torch.tensor(all_labels, dtype=torch.int64) 

        # Calculate class frequencies
        class_counts = torch.bincount(torch_labels)
        logging.info(f'Class distribution: {class_counts.tolist()}')
        
        # Calculate inverse class frequencies
        class_weights = 1.0 / class_counts.float()
        # Normalize weights
        class_weights /= class_weights.sum()
        logging.info(f'Class weights: {class_weights.tolist()}')

    # Create data loaders
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    logging.info(f'Data loaders created | Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')

    # Create model
    model = Model(args).to(device)
    logging.info(f'Model loaded on {device}')

    # Setup parameter groups for optimizer
    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximatorMulti.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
        logging.info('Created checkpoint directory: ./ckpt')

    # Create optimizer with parameter groups
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([{'params': base_param},
                            {'params': model.approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximatorMulti.parameters(), 'lr': args.lr / 2},
                            ],
                            lr=args.lr, weight_decay=args.weight_decay)
        logging.info(f'Adam optimizer created | LR: {args.lr} | Weight decay: {args.weight_decay}')
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD([{'params': base_param},
                            {'params': model.approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximatorMulti.parameters(), 'lr': args.lr / 2},
                            ], lr=args.lr, weight_decay=args.weight_decay)
        logging.info(f'SGD optimizer created | LR: {args.lr} | Weight decay: {args.weight_decay}')

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=0.1)
    logging.info(f'LR scheduler milestones: {args.scheduler_milestones}')

    # Loss functions
    class WeightedCrossEntropyLoss(torch.nn.Module):
        def __init__(self):
            super(WeightedCrossEntropyLoss, self).__init__()

        def forward(self, input, target, sample_weights):
            ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(input, target)
            weighted_ce_loss = ce_loss * sample_weights
            return torch.mean(weighted_ce_loss)

    if args.weights == 'Inverse':
        criterion = WeightedCrossEntropyLoss()
        logging.info('Using weighted cross-entropy loss')
    elif args.weights == 'Normal':
        criterion = torch.nn.CrossEntropyLoss()
        logging.info('Using standard cross-entropy loss')
    
    criterion2 = torch.nn.BCELoss()

    is_topk = True
    gt = np.load(args.gt)
    logging.info(f'Ground truth loaded: {gt.shape}')

    latestepoch = 0
    
    train_losses = []
    accuracy_arr = []
    f1_arr = []
    precision_arr = []
    recall_arr = []
    roc_auc_arr = []
    mAP_arr = []
    
    # Training loop
    try:
        for epoch in range(args.max_epoch - latestepoch):
            logging.info(f'\n{"="*80}')
            logging.info(f'EPOCH {epoch + 1 + latestepoch}/{args.max_epoch} | Mode: {args.online_mode} | Weights: {args.weights}')
            logging.info(f'{"="*80}\n')
            
            st = time.time()
            try:
                # Training phase
                if args.weights == 'Inverse':
                    loss = train(train_loader, model, optimizer, criterion, criterion2, device, is_topk, class_weights, args.online_mode, args)
                elif args.weights == 'Normal':
                    loss = train(train_loader, model, optimizer, criterion, criterion2, device, is_topk, None, args.online_mode, args)
                else:
                    logging.error(f'Unknown weights mode: {args.weights}')
                    break
                
                train_losses.append(loss)
                
                # Scheduler step after optimizer steps (PyTorch requirement)
                scheduler.step()
                
                elapsed = time.time() - st
                logging.info(f'Epoch training completed in {elapsed:.2f}s | Loss: {loss:.4f}')
                
            except Exception as e:
                logging.error(f'Error during training epoch {epoch + 1}: {str(e)}')
                continue
            
            # Save model checkpoint every 2 epochs
            try:
                if epoch % 2 == 0 and not epoch == 0:
                    model_path = f'./ckpt/{args.model_name}{epoch}.pkl'
                    torch.save(model.state_dict(), model_path)
                    logging.info(f'Model checkpoint saved: {model_path}')
            except Exception as e:
                logging.warning(f'Failed to save checkpoint: {str(e)}')

            # Testing phase
            try:
                roc_auc, f1, precision1, recall1, accuracy, mAP, cm = test(test_loader, model, device, gt, args.online_mode, args)
                accuracy_arr.append(accuracy)
                f1_arr.append(f1)
                precision_arr.append(precision1)
                recall_arr.append(recall1)
                roc_auc_arr.append(roc_auc)
                mAP_arr.append(mAP)
                
                logging.info(f'\nTest Metrics:')
                logging.info(f'  ROC AUC: {roc_auc:.4f}')
                logging.info(f'  mAP: {mAP:.4f}\n')
                logging.info(f'  F1 Score: {f1:.4f}')
                logging.info(f'  Precision: {precision1:.4f}')
                logging.info(f'  Recall: {recall1:.4f}')
                logging.info(f'  Accuracy: {accuracy:.4f}')
                
            except Exception as e:
                logging.error(f'Error during testing epoch {epoch + 1}: {str(e)}')
                continue
        
        # Save final results
        logging.info('\n' + "="*80)
        logging.info('Training completed. Saving results...')
        
        try:
            np.save(f'./ckpt/train_losses_{args.online_mode}_{args.weights}.npy', np.array(train_losses))
            np.save(f'./ckpt/roc_auc_{args.online_mode}_{args.weights}.npy', np.array(roc_auc_arr))
            np.save(f'./ckpt/f1_{args.online_mode}_{args.weights}.npy', np.array(f1_arr))
            np.save(f'./ckpt/precision_{args.online_mode}_{args.weights}.npy', np.array(precision_arr))
            np.save(f'./ckpt/recall_{args.online_mode}_{args.weights}.npy', np.array(recall_arr))
            np.save(f'./ckpt/accuracy_{args.online_mode}_{args.weights}.npy', np.array(accuracy_arr))
            np.save(f'./ckpt/mAP_{args.online_mode}_{args.weights}.npy', np.array(mAP_arr))
            
            # Save final model
            torch.save(model.state_dict(), f'./ckpt/{args.model_name}.pkl')
            logging.info('All results saved successfully!')
            
        except Exception as e:
            logging.error(f'Error saving results: {str(e)}')
    
    except KeyboardInterrupt:
        logging.info('\nTraining interrupted by user')
    except Exception as e:
        logging.error(f'Critical error in training loop: {str(e)}')
        raise

