from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, average_precision_score
import numpy as np
import torch
from tqdm import tqdm
import logging

def get_modality_slices(modality, feature_size):
	audio_dim = 128
	if feature_size is None:
		raise ValueError('feature_size is required to compute modality slices')

	if modality == 'AUDIO':
		return [('AUDIO', slice(0, feature_size))]
	elif modality == 'RGB':
		return [('RGB', slice(0, feature_size))]
	elif modality == 'FLOW':
		return [('FLOW', slice(0, feature_size))]
	elif modality == 'MIX':
		mid = feature_size // 2
		return [('RGB', slice(0, mid)), ('FLOW', slice(mid, feature_size))]
	elif modality == 'MIX2':
		rgb_dim = feature_size - audio_dim
		return [('RGB', slice(0, rgb_dim)), ('AUDIO', slice(rgb_dim, feature_size))]
	elif modality == 'MIX3':
		flow_dim = feature_size - audio_dim
		return [('FLOW', slice(0, flow_dim)), ('AUDIO', slice(flow_dim, feature_size))]
	elif modality == 'MIX_ALL':
		rem = feature_size - audio_dim
		first = rem // 2
		second = rem - first
		return [('RGB', slice(0, first)), ('FLOW', slice(first, first + second)), ('AUDIO', slice(feature_size - audio_dim, feature_size))]
	else:
		return [(modality, slice(0, feature_size))]


def compute_modality_drops(model, inputs, seq_len, modality_slices):
	full_logits, _ = model(inputs, seq_len)
	full_probs = torch.softmax(full_logits, dim=2).mean(dim=0)
	drops = {}
	if len(modality_slices) == 1:
		for name, _ in modality_slices:
			drops[name] = np.zeros((full_probs.size(0), full_probs.size(1)), dtype=np.float32)
		return full_probs.cpu().numpy(), drops

	for name, _ in modality_slices:
		ablated = inputs.clone()
		for other_name, sl in modality_slices:
			if other_name != name:
				ablated[..., sl] = 0.0
		ablated_logits, _ = model(ablated, seq_len)
		ablated_probs = torch.softmax(ablated_logits, dim=2).mean(dim=0)
		drops[name] = (full_probs - ablated_probs).cpu().numpy()

	return full_probs.cpu().numpy(), drops


def test(dataloader, model, device, gt, modality='MIX2', feature_size=None):
	with torch.no_grad():
		model.eval()
		all_preds = []
		all_probs = []
		all_drops = {}

		modality_slices = get_modality_slices(modality, feature_size)
		modality_names = [name for name, _ in modality_slices]
		for name in modality_names:
			all_drops[name] = []

		try:

			pbar = tqdm(dataloader, desc=f"Testing")

			for i, input in enumerate(pbar):
				try:
					input = input.to(device)
					full_probs_batch, drops_batch = compute_modality_drops(model, input, seq_len=None, modality_slices=modality_slices)
					
					# Per-class thresholds (tune these later)
					thresholds = np.array([0.5, 0.6, 0.7, 0.65, 0.5, 0.95, 0.65])

					batch_preds = []

					for probs in full_probs_batch:
						valid_classes = np.where(probs > thresholds)[0]

						if len(valid_classes) > 0:
							pred = valid_classes[np.argmax(probs[valid_classes])]
						else:
							pred = np.argmax(probs)

						batch_preds.append(pred)

					batch_preds = np.array(batch_preds).astype(np.float32)
					
					all_preds.append(batch_preds)
					all_probs.append(full_probs_batch)

					for name in modality_names:
						all_drops[name].append(drops_batch[name])

					pbar.set_postfix({
						'batch': f'{i+1}/{len(dataloader)}',
						'pred_shape': batch_preds.shape,
						'probs_shape': full_probs_batch.shape
					})
				except Exception as e:
					logging.error(f"[TEST] Error processing batch {i}: {str(e)}")
					continue

			all_preds = np.concatenate(all_preds, axis=0) if len(all_preds) > 0 else np.array([], dtype=np.float32)
			all_probs = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else np.zeros((0, 7), dtype=np.float32)
			for name in modality_names:
				all_drops[name] = np.concatenate(all_drops[name], axis=0) if len(all_drops[name]) > 0 else np.zeros((0, all_probs.shape[1] if all_probs.size else 7), dtype=np.float32)

			logging.info(f'Predictions shape: {all_preds.shape}')
			logging.info(f'Probabilities shape: {all_probs.shape}')
			logging.info(f'Ground truth shape: {len(gt)}')
			logging.info(f'ground truth sample (first 100): {gt[:100]}')
			logging.info(f'Sample predictions (first 100): {all_preds[:100]}')

			repeated_probs = np.repeat(all_probs, 16, axis=0)
			repeated_preds = np.repeat(all_preds, 16)
			repeated_drops = {name: np.repeat(all_drops[name], 16, axis=0) for name in modality_names}
			gt_array = np.array(gt)

			if len(gt_array) != len(repeated_preds):
				min_len = min(len(gt_array), len(repeated_preds))
				logging.warning(f'Ground truth length ({len(gt_array)}) does not match repeated predictions length ({len(repeated_preds)}). Trimming to {min_len} values.')
				gt_array = gt_array[:min_len]
				repeated_probs = repeated_probs[:min_len]
				repeated_preds = repeated_preds[:min_len]
				for name in modality_names:
					repeated_drops[name] = repeated_drops[name][:min_len]

			try:
				roc_auc = roc_auc_score(gt_array.tolist(), repeated_probs, multi_class='ovr', average='weighted')
			except Exception as e:
				logging.warning(f'ROC AUC calculation failed: {str(e)}, using 0.0')
				roc_auc = 0.0

			f1 = f1_score(gt_array.tolist(), repeated_preds, average='weighted', zero_division=0)
			prec = precision_score(gt_array.tolist(), repeated_preds, average='weighted', zero_division=0)
			recal = recall_score(gt_array.tolist(), repeated_preds, average='weighted', zero_division=0)
			acc = accuracy_score(gt_array.tolist(), repeated_preds)
			cm = confusion_matrix(gt_array.tolist(), repeated_preds)

			try:
				mAP = average_precision_score(gt_array.tolist(), repeated_probs, average='weighted')
			except Exception as e:
				logging.warning(f'mAP calculation failed: {str(e)}, using 0.0')
				mAP = 0.0

			num_classes = repeated_probs.shape[1]
			avg_modality_drops = np.zeros((num_classes, len(modality_names)), dtype=np.float32)
			for class_id in range(num_classes):
				indices = np.where(gt_array == class_id)[0]
				for mod_idx, name in enumerate(modality_names):
					if indices.size > 0:
						avg_modality_drops[class_id, mod_idx] = np.mean(repeated_drops[name][indices, class_id])
					else:
						avg_modality_drops[class_id, mod_idx] = 0.0

			top_modalities = [modality_names[int(np.argmax(avg_modality_drops[class_id]))] for class_id in range(num_classes)]

			logging.info(f'Modality names: {modality_names}')
			for class_id, name in enumerate(top_modalities):
				logging.info(f'Class {class_id} top modality: {name}')

			target_names = ['Normal', 'Fighting', 'Shooting', 'Explosion', 'Riot', 'Abuse', 'Car accident']
			report = classification_report(gt_array.tolist(), repeated_preds, target_names=target_names, zero_division=0)

			return roc_auc, f1, prec, recal, acc, mAP, cm, report, avg_modality_drops, top_modalities
		except Exception as e:
			logging.error(f"[TEST] Critical error in test function: {str(e)}")
			return 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, np.zeros((0, 0), dtype=np.float32), []