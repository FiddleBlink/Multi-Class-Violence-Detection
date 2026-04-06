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
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of correct class

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

def setup_logging(log_dir):
	"""Setup logging with both console and file handlers"""
	# Create logs directory if it doesn't exist
	os.makedirs(log_dir, exist_ok=True)
	
	# Get root logger
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	
	# Clear any existing handlers
	logger.handlers.clear()
	
	# Create formatters and handlers
	formatter = logging.Formatter('[%(levelname)s] %(message)s')
	
	# Console handler
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	
	# File handler
	log_file = os.path.join(log_dir, 'training.log')
	file_handler = logging.FileHandler(log_file)
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	
	return logger

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
	
	# Create checkpoint and logs directories, then setup logging
	if not os.path.exists(f'./ckpt/{args.model_name}'):
		os.makedirs(f'./ckpt/{args.model_name}')
	
	log_dir = f'./ckpt/{args.model_name}/logs'
	setup_logging(log_dir)
	
	logging.info(f'=== XDVioDet Training Started ===')
	logging.info(f'Mode: {args.scoring_mode} | Weights: {args.weights} | Optimizer: {args.optimizer}')
	logging.info(f'LR: {args.lr} | Batch Size: {args.batch_size} | Max Epochs: {args.max_epoch}')
	logging.info(f'Logging to: {log_dir}')

	device = torch.device("cuda")
	logging.info(f'Device: {device}')
	class_weights = [1.0, 1.1216551065444946, 1.228792428970337, 1.264064073562622, 1.1746114492416382, 6.067634582519531, 1.1627792119979858]

	if args.weights == 'Inverse':
		if(class_weights is not None):
			class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
			logging.info(f'Using inverse class weights: {class_weights.cpu().numpy()}')
		else:
			logging.info('Calculating inverse class weights...')
			train_data_all = DataLoader(Dataset(args, test_mode=False), batch_size=args.batch_size, shuffle=True)
			all_labels = []

			for i, (input, label) in enumerate(train_data_all):
				print(i)
				all_labels.append(label)

			all_labels = torch.cat(all_labels, dim=0)
			print(f'DataLoader: {all_labels.shape}')

			# torch_labels = torch.tensor(all_labels, dtype=torch.int64) 
			torch_labels = all_labels.clone().detach().to(torch.int64) 

			# Calculate class frequencies
			class_counts = torch.bincount(torch_labels)
			# Calculate inverse class frequencies
			beta = 0.999
			effective_num = 1.0 - torch.pow(beta, class_counts.float())
			class_weights = (1.0 - beta) / effective_num
			class_weights = class_weights / class_weights.min()
			logging.info(f'Class weights: {class_weights.tolist()}')
	

	train_loader = DataLoader(Dataset(args, test_mode=False),
							  batch_size=args.batch_size, shuffle=True,
							  num_workers=args.workers, pin_memory=True)
	test_loader = DataLoader(Dataset(args, test_mode=True),
							  batch_size=5, shuffle=False,
							  num_workers=args.workers, pin_memory=True)
	logging.info(f'Data loaders created | Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')


	device = torch.device('cuda:{}'.format(args.gpus) if args.gpus != '-1' else 'cpu')
	model = Model(args).to(device)
	logging.info(f'Model loaded on {device}')

	# for name, value in model.named_parameters():
	#     print(name)
	approximator_param = list(map(id, model.approximator.parameters()))
	approximator_param += list(map(id, model.conv1d_approximator.parameters()))
	base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

	if args.optimizer == 'Adam':
		optimizer = optim.Adam([{'params': base_param},
							{'params': model.approximator.parameters(), 'lr': args.lr / 2},
							{'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
							],
							lr=args.lr, weight_decay=0.000)
		logging.info(f'Adam optimizer created | LR: {args.lr}')
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD([{'params': base_param},
							{'params': model.approximator.parameters(), 'lr': args.lr / 2},
							{'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
							], lr=args.lr, weight_decay=0.000)
		logging.info(f'SGD optimizer created | LR: {args.lr}')

	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

	# class WeightedCrossEntropyLoss(torch.nn.Module):
	# 	def __init__(self):
	# 		super(WeightedCrossEntropyLoss, self).__init__()

	# 	def forward(self, input, target, sample_weights):
	# 		ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(input, target)
	# 		weighted_ce_loss = ce_loss * sample_weights
	# 		return torch.mean(weighted_ce_loss)

	

	if args.weights == 'Inverse':
		# criterion = WeightedCrossEntropyLoss()
		# criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
		criterion = FocalLoss(alpha=class_weights.to(device), gamma=1.5)
		logging.info('Using weighted cross-entropy loss')
	elif args.weights == 'Normal':
		# criterion = torch.nn.CrossEntropyLoss()
		criterion = FocalLoss(alpha=None, gamma=1.5)
		logging.info('Using standard cross-entropy loss')
	criterion2 = torch.nn.BCEWithLogitsLoss()

	is_topk = False
	gt = np.load(args.gt)
	logging.info(f'Ground truth loaded: {gt.shape}')

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
	mAP_arr = []
	cm_arr = []
	report_arr = []
	
	try:
		for epoch in range(args.max_epoch - latestepoch):
			logging.info(f'\n{"="*80}')
			logging.info(f'EPOCH {epoch + 1 + latestepoch}/{args.max_epoch} | Mode: {args.scoring_mode} | Weights: {args.weights}')
			logging.info(f'{"="*80}\n')
			
			st = time.time()
			try:
				if args.weights == 'Inverse':
					loss = train(train_loader, model, optimizer, criterion, criterion2, device, is_topk, class_weights, args.scoring_mode)
				elif args.weights == 'Normal':
					loss = train(train_loader, model, optimizer, criterion, criterion2, device, is_topk, None, args.scoring_mode)
				
				train_losses.append(loss)
				
				scheduler.step()
				elapsed = time.time() - st
				logging.info(f'Epoch training completed in {elapsed:.2f}s | Loss: {loss:.4f}')
			
			except Exception as e:
				logging.error(f'Error during training epoch {epoch + 1}: {str(e)}')
				continue
			
			if epoch % 2 == 0 and not epoch == 0:
				torch.save(model.state_dict(), f'./ckpt/{args.model_name}/'+args.model_name+'{}.pkl'.format(epoch))
				logging.info(f'Model checkpoint saved')

			roc_auc, f1, precision1, recall1, accuracy, mAP, cm, report, avg_modality_drops, top_modalities = test(test_loader, model, device, gt, args.modality, args.feature_size)
			# print('Epoch {0}/{1}: offline roc_auc:{2:.4}'.format(epoch, args.max_epoch, roc_auc))
			accuracy_arr.append(accuracy)
			f1_arr.append(f1)
			precision_arr.append(precision1)
			recall_arr.append(recall1)
			roc_auc_arr.append(roc_auc)
			mAP_arr.append(mAP)
			cm_arr.append(cm)
			report_arr.append(report)
			logging.info(f'Per-category modality contribution matrix shape: {avg_modality_drops.shape}')
			logging.info(f'Per-category top modality mapping: {top_modalities}')
	
			logging.info(f'\nTest Metrics:')
			logging.info(f'  ROC AUC: {roc_auc:.4f}')
			logging.info(f'  mAP: {mAP:.4f}\n')
			logging.info(f'  F1 Score: {f1:.4f}')
			logging.info(f'  Precision: {precision1:.4f}')
			logging.info(f'  Recall: {recall1:.4f}')
			logging.info(f'  Accuracy: {accuracy:.4f}')
			logging.info(f'\nClassification Report:\n{report}')
			logging.info(f'\nConfusion Matrix:\n{cm}')
		
		logging.info('\n' + "="*80)
		logging.info('Training completed. Saving results...')

		np.save(f'./ckpt/{args.model_name}/train_losses_{args.scoring_mode}_{args.weights}.npy', np.array(train_losses))
		np.save(f'./ckpt/{args.model_name}/roc_auc_{args.scoring_mode}_{args.weights}.npy', np.array(roc_auc_arr))
		np.save(f'./ckpt/{args.model_name}/f1_{args.scoring_mode}_{args.weights}.npy', np.array(f1_arr))
		np.save(f'./ckpt/{args.model_name}/precision_{args.scoring_mode}_{args.weights}.npy', np.array(precision_arr))
		np.save(f'./ckpt/{args.model_name}/recall_{args.scoring_mode}_{args.weights}.npy', np.array(recall_arr))
		np.save(f'./ckpt/{args.model_name}/accuracy_{args.scoring_mode}_{args.weights}.npy', np.array(accuracy_arr))
		np.save(f'./ckpt/{args.model_name}/mAP_{args.scoring_mode}_{args.weights}.npy', np.array(mAP_arr))
		np.save(f'./ckpt/{args.model_name}/cm_{args.scoring_mode}_{args.weights}.npy', np.array(cm_arr, dtype=object))
		np.save(f'./ckpt/{args.model_name}/report_{args.scoring_mode}_{args.weights}.npy', np.array(report_arr, dtype=object))
		np.save(f'./ckpt/{args.model_name}/modality_contribution_{args.scoring_mode}_{args.weights}.npy', avg_modality_drops)
		np.save(f'./ckpt/{args.model_name}/modality_top_{args.scoring_mode}_{args.weights}.npy', np.array(top_modalities, dtype=object))

		torch.save(model.state_dict(), f'./ckpt/{args.model_name}/' + args.model_name + '.pkl')
	
	except KeyboardInterrupt:
		logging.info('\nTraining interrupted by user')
	except Exception as e:
		logging.error(f'Critical error in training loop: {str(e)}')
		raise
