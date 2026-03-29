from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, average_precision_score
import numpy as np
import torch
from tqdm import tqdm
import logging

def test(dataloader, model, device, gt):
	with torch.no_grad():
		model.eval()
		all_preds = []
		all_probs = np.zeros((0, 7))
		
		try:

			pbar = tqdm(dataloader, desc=f"Testing")

			for i, input in enumerate(pbar):
				try:
					input = input.to(device)
					logits, logits2 = model(inputs=input, seq_len=None)
					probs = torch.softmax(logits, 2)
					probs = torch.mean(probs, dim=0)  # Fix: Replace 0 with dim=0
					pred = torch.argmax(probs, 1).float()
					all_preds.extend(pred.cpu().numpy())
					all_probs = np.concatenate((all_probs, probs.view(-1, probs.size(-1)).cpu().numpy()))

					pbar.set_postfix({
                        'batch': f'{i+1}/{len(dataloader)}',
                        'pred_shape': pred.shape,
                        'probs_shape': probs.shape
                    })
				except Exception as e:
					logging.error(f"[TEST] Error processing batch {i}: {str(e)}")
					continue

			all_preds = np.array(all_preds)

			logging.info(f'Predictions shape: {all_preds.shape}')
			logging.info(f'Probabilities shape: {all_probs.shape}')
			logging.info(f'Ground truth shape: {len(gt)}')
			logging.info(f'Sample predictions (first 100): {all_preds[:100]}')

			# pred: 145649
			# GT: 2330384
			repeated_probs = np.repeat(all_probs, 16, axis=0)
			repeated_preds = np.repeat(all_preds, 16)

			try:
				roc_auc = roc_auc_score(list(gt), repeated_probs, multi_class='ovr', average='weighted')
			except Exception as e:
				logging.warning(f'ROC AUC calculation failed: {str(e)}, using 0.0')
				roc_auc = 0.0

			f1 = f1_score(list(gt), repeated_preds, average='weighted', zero_division=0)
			prec = precision_score(list(gt), repeated_preds, average='weighted', zero_division=0)
			recal = recall_score(list(gt), repeated_preds, average='weighted', zero_division=0)
			acc = accuracy_score(list(gt), repeated_preds)
			cm = confusion_matrix(list(gt), repeated_preds)

			try:
				mAP = average_precision_score(list(gt), repeated_probs, average='weighted')
			except Exception as e:
				logging.warning(f'mAP calculation failed: {str(e)}, using 0.0')
				mAP = 0.0

			target_names = ['Normal', 'Fighting', 'Shooting', 'Explosion', 'Riot', 'Abuse', 'Car accident']
			report = classification_report(list(gt), repeated_preds, target_names=target_names, zero_division=0)

			# logging.info(f'\n=== Test Results ===')
			# logging.info(f'ROC AUC: {roc_auc:.4f}')
			# logging.info(f'F1 Score: {f1:.4f}')
			# logging.info(f'Precision: {prec:.4f}')
			# logging.info(f'Recall: {recal:.4f}')
			# logging.info(f'Accuracy: {acc:.4f}')
			# logging.info(f'mAP: {mAP:.4f}')
			# logging.info(f'\nClassification Report:\n{report}')
			# logging.info(f'\nConfusion Matrix:\n{cm}')

			return roc_auc, f1, prec, recal, acc, mAP, cm, report
		except Exception as e:
			logging.error(f"[TEST] Critical error in test function: {str(e)}")
			return 0.0, 0.0, 0.0, 0.0, 0.0, None