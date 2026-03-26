import torch
import logging
import time
from tqdm import tqdm


def CLAS(logits, label, seq_len, criterion, device, sample_weights=None, is_topk=True):
    """Classification loss for multi-class violence detection"""
    logits = logits.squeeze()
    instance_logits = torch.zeros(7).to(device)
    outx = []
    for i in range(logits.shape[0]):
        if is_topk:
            weights = torch.softmax(logits[i][:seq_len[i]] / 0.5, dim=0)  # temperature
            tmp = torch.sum(weights * logits[i][:seq_len[i]], dim=0)
        else:
            tmp = torch.mean(logits[i][:seq_len[i]], dim=0)
        outx.append(tmp)
    instance_logits = torch.stack(outx)

    if sample_weights is None:
        clsloss = criterion(instance_logits, label)
    else:
        clsloss = criterion(instance_logits, label, sample_weights)
    return clsloss


def CLAS2(logits, label, seq_len, criterion, device, is_topk=True):
    """Binary classification loss for violence detection"""
    logits = logits.squeeze()
    outx = []
    for i in range(logits.shape[0]):
        if is_topk:
            # Aggregate multi-class logits to binary using weighted combination
            weights = torch.softmax(logits[i][:seq_len[i]] / 0.5, dim=0)  # temperature
            tmp = torch.mean(torch.sum(weights * logits[i][:seq_len[i]], dim=1))  # Average over classes
        else:
            # Simple mean aggregation
            tmp = torch.mean(logits[i, :seq_len[i]])
        outx.append(tmp)
    
    instance_logits = torch.stack(outx)
    instance_logits = torch.sigmoid(instance_logits)
    clsloss = criterion(instance_logits, label.float())
    return clsloss


def CENTROPY(logits, logits2, seq_len, device, online_mode, sample_weights=None):
    """Cross-entropy loss between teacher and student predictions"""
    instance_logits = 0.0

    if online_mode == 'Binary':
        for i in range(logits.shape[0]):
            tmp1 = torch.softmax(logits[i, :seq_len[i]], dim=0)
            tmp1 = torch.mean(tmp1, dim=1)
            tmp1 = tmp1.squeeze()
            tmp2 = torch.softmax(logits2[i, :seq_len[i]], dim=0).squeeze()
            crosOut = -torch.mean(tmp1.detach() * torch.log(tmp2 + 1e-8))
            if sample_weights is not None:
                instance_logits += crosOut * sample_weights[i]
            else:
                instance_logits += crosOut
    elif online_mode == 'Multi':
        for i in range(logits.shape[0]):
            tmp1 = torch.softmax(logits[i, :seq_len[i]], dim=0)
            tmp1 = torch.mean(tmp1, dim=1)
            tmp1 = tmp1.squeeze()
            tmp2 = torch.softmax(logits2[i, :seq_len[i]], dim=0)
            tmp2 = torch.mean(tmp2, dim=1)
            tmp2 = tmp2.squeeze()
            crosOut = -torch.mean(tmp1.detach() * torch.log(tmp2 + 1e-8))
            if sample_weights is not None:
                instance_logits += crosOut * sample_weights[i]
            else:
                instance_logits += crosOut

    instance_logits = instance_logits / logits.shape[0]
    return instance_logits

def temporal_smoothness(logits, seq_len):
    loss = 0
    for i in range(logits.shape[0]):
        diff = logits[i, 1:seq_len[i]] - logits[i, :seq_len[i]-1]
        loss += torch.mean(diff ** 2)
    return loss / logits.shape[0]

def train(dataloader, model, optimizer, criterion, criterion2, device, is_topk, class_weights, online_mode, args=None):
    """Enhanced training function with better logging and error handling"""
    model.train()
    total_epoch_loss = 0
    total_clsloss = 0
    total_clsloss2 = 0
    total_croloss = 0

    # Setup progress bar if tqdm is available
    try:
        pbar = tqdm(dataloader, desc=f"Training (Mode: {online_mode})")
        use_tqdm = True
    except ImportError:
        pbar = dataloader
        use_tqdm = False

    start_time = time.time()

    for i, (input_data, label) in enumerate(pbar):
        try:
            # Compute sequence lengths
            seq_len = torch.sum(torch.max(torch.abs(input_data), dim=2)[0] > 0, 1)
            input_data = input_data[:, :torch.max(seq_len), :]

            # Move to device
            input_data, label = input_data.float().to(device), label.float().to(device)
            label = label.to(torch.int64)

            # Handle class weights
            if class_weights is not None:
                class_weights = class_weights.to(device)
                sample_weights = class_weights[label]
            else:
                sample_weights = None

            # Forward pass
            logits, logits2 = model(input_data, seq_len)

            # Compute losses
            clsloss = CLAS(logits, label, seq_len, criterion, device, sample_weights, is_topk)

            if online_mode == 'Binary':
                label2 = torch.where(label >= 1, torch.tensor(1).to(device), label)
                clsloss2 = CLAS2(logits2, label2, seq_len, criterion2, device, is_topk)
            elif online_mode == 'Multi':
                clsloss2 = CLAS(logits2, label, seq_len, criterion, device, sample_weights, is_topk)

            croloss = CENTROPY(logits, logits2, seq_len, device, online_mode, sample_weights)

            # Combine losses (croloss coefficient could be made configurable)
            croloss_weight = getattr(args, 'croloss_weight', 5.0) if args else 5.0
            total_loss = clsloss + clsloss2 + croloss_weight * croloss

            smooth_loss = temporal_smoothness(logits, seq_len)
            total_loss += 0.1 * smooth_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping if specified
            if args and hasattr(args, 'grad_clip') and args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            # Accumulate losses
            total_epoch_loss += total_loss.item()
            total_clsloss += clsloss.item()
            total_clsloss2 += clsloss2.item()
            total_croloss += croloss.item()

            # Update progress bar
            if use_tqdm:
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'cls1': f'{clsloss.item():.4f}',
                    'cls2': f'{clsloss2.item():.4f}',
                    'cro': f'{croloss.item():.4f}'
                })

        except Exception as e:
            print(f"[ERROR] Batch {i} failed: {str(e)}")
            continue

    # Calculate averages
    num_batches = len(dataloader)
    avg_loss = total_epoch_loss / num_batches
    avg_clsloss = total_clsloss / num_batches
    avg_clsloss2 = total_clsloss2 / num_batches
    avg_croloss = total_croloss / num_batches

    epoch_time = time.time() - start_time

    # Log final statistics
    print(f"[TRAIN] Epoch completed in {epoch_time:.2f}s")
    print(f"[TRAIN] Average Loss: {avg_loss:.4f} (CLS1: {avg_clsloss:.4f}, CLS2: {avg_clsloss2:.4f}, CRO: {avg_croloss:.4f})")

    return avg_loss