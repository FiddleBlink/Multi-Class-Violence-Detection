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

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

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
	
	logging.info(f'=== XDVioDet Training Started ===')
	logging.info(f'Mode: {args.scoring_mode} | Weights: {args.weights} | Optimizer: {args.optimizer}')
	logging.info(f'LR: {args.lr} | Batch Size: {args.batch_size} | Max Epochs: {args.max_epoch}')

	device = torch.device("cuda")
	logging.info(f'Device: {device}')

	if args.weights == 'Inverse':
		logging.info('Calculating inverse class weights...')
		train_data_all = DataLoader(Dataset(args, test_mode=False), batch_size=args.batch_size, shuffle=True)
		all_labels = []

		for i, (input, label) in enumerate(train_data_all):
			print(i)
			all_labels.append(label)

		all_labels = torch.cat(all_labels, dim=0)
		print(f'DataLoader: {all_labels.shape}')

		torch_labels = torch.tensor(all_labels, dtype=torch.int64) 

		# Calculate class frequencies
		class_counts = torch.bincount(torch_labels)
		# Calculate inverse class frequencies
		class_weights = 1.0 / class_counts
		# Normalize weights
		class_weights /= class_weights.sum()
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

	if not os.path.exists('./ckpt'):
		os.makedirs('./ckpt')
		logging.info('Created checkpoint directory: ./ckpt')

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
				torch.save(model.state_dict(), './ckpt/'+args.model_name+'{}.pkl'.format(epoch))
				logging.info(f'Model checkpoint saved')

			roc_auc, f1, precision1, recall1, accuracy, mAP = test(test_loader, model, device, gt)
			# print('Epoch {0}/{1}: offline roc_auc:{2:.4}'.format(epoch, args.max_epoch, roc_auc))
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
		
		logging.info('\n' + "="*80)
		logging.info('Training completed. Saving results...')

		np.save(f'./ckpt/train_losses_{args.scoring_mode}_{args.weights}.npy', np.array(train_losses))
		np.save(f'./ckpt/roc_auc_{args.scoring_mode}_{args.weights}.npy', np.array(roc_auc_arr))
		np.save(f'./ckpt/f1_{args.scoring_mode}_{args.weights}.npy', np.array(f1_arr))
		np.save(f'./ckpt/precision_{args.scoring_mode}_{args.weights}.npy', np.array(precision_arr))
		np.save(f'./ckpt/recall_{args.scoring_mode}_{args.weights}.npy', np.array(recall_arr))
		np.save(f'./ckpt/accuracy_{args.scoring_mode}_{args.weights}.npy', np.array(accuracy_arr))
		np.save(f'./ckpt/mAP_{args.online_mode}_{args.weights}.npy', np.array(mAP_arr))

		torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
	
	except KeyboardInterrupt:
		logging.info('\nTraining interrupted by user')
	except Exception as e:
		logging.error(f'Critical error in training loop: {str(e)}')
		raise
