import torch
import logging
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def CLAS(logits, label, seq_len, criterion, device, sample_weights=None, is_topk=True):
    logits = logits.squeeze()
    outx = []
    for i in range(logits.shape[0]):
        if is_topk:
            k = int(seq_len[i] // 16 + 1)
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=k, dim=0, largest=True)
            tmp = torch.mean(tmp, dim=0)
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
    logits = logits.squeeze()
    instance_logits = []

    for i in range(logits.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            tmp = torch.mean(tmp).view(1)
        else:
            tmp = torch.mean(logits[i, :seq_len[i]]).view(1)

        instance_logits.append(tmp)

    instance_logits = torch.cat(instance_logits)

    # ❌ REMOVE sigmoid (important for AMP)
    # instance_logits = torch.sigmoid(instance_logits)

    clsloss = criterion(instance_logits, label.float())  # BCEWithLogitsLoss

    return clsloss


def CENTROPY(logits, logits2, seq_len, device, scoring_mode, sample_weights=None):
    instance_logits = 0.0

    if scoring_mode == 'Binary':
        for i in range(logits.shape[0]):
            tmp1 = torch.softmax(logits[i, :seq_len[i]], dim=0)
            tmp1 = torch.mean(tmp1, dim=1).squeeze()

            # Use sigmoid here (OK for loss alignment, not BCE)
            tmp2 = torch.sigmoid(logits2[i, :seq_len[i]]).squeeze()

            crosOut = -torch.mean(tmp1.detach() * torch.log(tmp2 + 1e-8))

            if sample_weights is not None:
                instance_logits += crosOut * sample_weights[i]
            else:
                instance_logits += crosOut

    elif scoring_mode == 'Multi':
        for i in range(logits.shape[0]):
            tmp1 = torch.softmax(logits[i, :seq_len[i]], dim=1)
            tmp1 = torch.mean(tmp1, dim=0).squeeze()

            tmp2 = torch.softmax(logits2[i, :seq_len[i]], dim=1)
            tmp2 = torch.mean(tmp2, dim=0).squeeze()

            crosOut = -torch.mean(tmp1.detach() * torch.log(tmp2 + 1e-8))

            if sample_weights is not None:
                instance_logits += crosOut * sample_weights[i]
            else:
                instance_logits += crosOut

    instance_logits = instance_logits / logits.shape[0]

    return instance_logits


def train(dataloader, model, optimizer, criterion, criterion2, device,
          is_topk, class_weights, scoring_mode):

    scaler = GradScaler()  # 🔥 AMP scaler

    model.train()
    total_epoch_loss = 0
    total_clsloss = 0
    total_clsloss2 = 0
    total_croloss = 0

    pbar = tqdm(dataloader, desc=f"Training (Mode: {scoring_mode})")

    start_time = time.time()

    for i, (input, label) in enumerate(pbar):
        try:
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0] > 0, 1)
            input = input[:, :torch.max(seq_len), :]

            input = input.float().to(device)
            label = label.to(torch.int64).to(device)

            if class_weights is not None:
                class_weights = class_weights.to(device)
                sample_weights = class_weights[label]
            else:
                sample_weights = None

            optimizer.zero_grad()

            # 🔥 AMP forward
            with autocast():
                logits, logits2 = model(input, seq_len)

                clsloss = CLAS(logits, label, seq_len, criterion,
                               device, sample_weights, is_topk)

                if scoring_mode == 'Binary':
                    label2 = torch.where(label >= 1,
                                         torch.tensor(1).to(device),
                                         torch.tensor(0).to(device))
                    clsloss2 = CLAS2(logits2, label2, seq_len,
                                     criterion2, device, is_topk)

                elif scoring_mode == 'Multi':
                    clsloss2 = CLAS(logits2, label, seq_len,
                                    criterion, device, sample_weights, is_topk)

                croloss = CENTROPY(logits, logits2, seq_len,
                                   device, scoring_mode, sample_weights)

                total_loss = clsloss + clsloss2 + 5 * croloss

            # 🔥 AMP backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += total_loss.item()
            total_clsloss += clsloss.item()
            total_clsloss2 += clsloss2.item()
            total_croloss += croloss.item()

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'cls1': f'{clsloss.item():.4f}',
                'cls2': f'{clsloss2.item():.4f}',
                'cro': f'{croloss.item():.4f}'
            })

        except Exception as e:
            logging.error(f"[ERROR] Batch {i} failed: {str(e)}")
            continue

    num_batches = len(dataloader)

    logging.info(f"[TRAIN] Epoch completed in {time.time() - start_time:.2f}s")
    logging.info(f"[TRAIN] Average Loss: {total_epoch_loss / num_batches:.4f}")

    return total_epoch_loss / num_batches