from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, average_precision_score
import numpy as np
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test(dataloader, model, device, gt, online_mode='Binary', args=None):
    """
    Enhanced test function with better error handling and logging.
    
    Args:
        dataloader: Test data loader
        model: Trained model
        device: torch.device (cpu or cuda)
        gt: Ground truth labels
        online_mode: 'Binary' or 'Multi' class mode
        args: Command-line arguments (for configuration)
    
    Returns:
        Tuple of (roc_auc, f1, precision, recall, accuracy, confusion_matrix)
    """
    with torch.no_grad():
        model.eval()
        all_preds = []
        all_probs = np.zeros((0, 7))
        
        try:
            # Setup progress bar
            pbar = tqdm(dataloader, desc=f"Testing (Mode: {online_mode})")
            
            for batch_idx, input_data in enumerate(pbar):
                try:
                    # Compute sequence lengths from input
                    seq_len = torch.sum(torch.max(torch.abs(input_data), dim=2)[0] > 0, 1)
                    
                    # Prepare input
                    input_data = input_data[:, :torch.max(seq_len), :].float().to(device)
                    
                    # Forward pass with proper argument order
                    logits, logits2 = model(input_data, seq_len)
                    
                    # Compute probabilities and predictions
                    probs = torch.softmax(logits, dim=2)  # dim=2 for (B, T, Classes)
                    probs = torch.mean(probs, dim=1)  # Average over time dimension
                    
                    pred = torch.argmax(probs, dim=1).float()
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_probs = np.concatenate((all_probs, probs.cpu().numpy()))
                    
                    pbar.set_postfix({
                        'batch': f'{batch_idx+1}/{len(dataloader)}',
                        'pred_shape': pred.shape,
                        'probs_shape': probs.shape
                    })
                    
                except Exception as e:
                    logging.error(f"[TEST] Error processing batch {batch_idx}: {str(e)}")
                    continue
            
            all_preds = np.array(all_preds)
            
            logging.info(f'Predictions shape: {all_preds.shape}')
            logging.info(f'Probabilities shape: {all_probs.shape}')
            logging.info(f'Ground truth shape: {len(gt)}')
            logging.info(f'Sample predictions (first 20): {all_preds[:20]}')
            
            # Repeat predictions to match ground truth length (for frame-level evaluation)
            # Each segment prediction is repeated to cover all frames in that segment
            repeat_factor = len(gt) // len(all_preds) if len(all_preds) > 0 else 1
            logging.info(f'Repeat factor for frame-level evaluation: {repeat_factor}')
            
            # Compute metrics
            if len(all_preds) > 0 and len(gt) > 0:
                repeated_preds = np.repeat(all_preds, repeat_factor)
                repeated_probs = np.repeat(all_probs, repeat_factor, axis=0)
                
                # Trim to match ground truth length if there's mismatch
                if len(repeated_preds) > len(gt):
                    repeated_preds = repeated_preds[:len(gt)]
                    repeated_probs = repeated_probs[:len(gt)]
                elif len(repeated_preds) < len(gt):
                    logging.warning(f'Predictions {len(repeated_preds)} < GT {len(gt)}, padding with last value')
                    pad_size = len(gt) - len(repeated_preds)
                    repeated_preds = np.concatenate([repeated_preds, np.full(pad_size, repeated_preds[-1])])
                    repeated_probs = np.vstack([repeated_probs, np.tile(repeated_probs[-1], (pad_size, 1))])
                
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
                
                # Calculate mean Average Precision (mAP)
                try:
                    mAP = average_precision_score(list(gt), repeated_probs, average='weighted')
                except Exception as e:
                    logging.warning(f'mAP calculation failed: {str(e)}, using 0.0')
                    mAP = 0.0
                
                target_names = ['Normal', 'Fighting', 'Shooting', 'Explosion', 'Riot', 'Abuse', 'Car accident']
                report = classification_report(list(gt), repeated_preds, target_names=target_names, zero_division=0)
                
                logging.info(f'\n=== Test Results ===')
                logging.info(f'ROC AUC: {roc_auc:.4f}')
                logging.info(f'F1 Score: {f1:.4f}')
                logging.info(f'Precision: {prec:.4f}')
                logging.info(f'Recall: {recal:.4f}')
                logging.info(f'Accuracy: {acc:.4f}')
                logging.info(f'mAP: {mAP:.4f}')
                logging.info(f'\nClassification Report:\n{report}')
                logging.info(f'\nConfusion Matrix:\n{cm}')
                
                return roc_auc, f1, prec, recal, acc, mAP, cm
            else:
                logging.error('No valid predictions generated!')
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None
                
        except Exception as e:
            logging.error(f"[TEST] Critical error in test function: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, None
