import torch


def CLAS(logits, label, seq_len, criterion, device, sample_weights=None, is_topk=True):
    logits = logits.squeeze()
    instance_logits = torch.zeros(7).to(device)  # tensor([])
    outx = []
    for i in range(logits.shape[0]):
        if is_topk:
            k = int(seq_len[i] // 16 + 1)  # Calculate k 
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=k, dim=0, largest=True) 
            tmp = torch.mean(tmp, dim=0)  # Average across the top-k frames
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
    instance_logits = torch.zeros(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            tmp = torch.mean(tmp).view(1)
        else:
            tmp = torch.mean(logits[i, :seq_len[i]]).view(1)
        instance_logits = torch.cat((instance_logits, tmp))
    
    instance_logits = torch.sigmoid(instance_logits)

    clsloss = criterion(instance_logits, label.float())
    return clsloss

def CENTROPY(logits, logits2, seq_len, device, online_mode, sample_weights=None):
    instance_logits = 0.0  # tensor([])

    if online_mode == 'Binary':
        for i in range(logits.shape[0]):
            tmp1 = torch.softmax(logits[i, :seq_len[i]], dim=0)
            tmp1 = torch.mean(tmp1, dim=1)  # Average across dim=1
            tmp1 = tmp1.squeeze()  
            tmp2 = torch.softmax(logits2[i, :seq_len[i]], dim=0).squeeze()
            crosOut = -torch.mean(tmp1.detach() * torch.log(tmp2))
            if sample_weights is not None:
                instance_logits += crosOut * sample_weights[i]
            else:
                instance_logits += crosOut
    elif online_mode == 'Multi':
        for i in range(logits.shape[0]):
            tmp1 = torch.softmax(logits[i, :seq_len[i]], dim=0)
            tmp1 = torch.mean(tmp1, dim=1)  # Average across dim=1
            tmp1 = tmp1.squeeze()  
            # tmp2 = torch.softmax(logits2[i, :seq_len[i]], dim=0).squeeze()
            tmp2 = torch.softmax(logits2[i, :seq_len[i]], dim=0)
            tmp2 = torch.mean(tmp2, dim=1)  # Average across dim=1
            tmp2 = tmp2.squeeze()  
            crosOut = -torch.mean(tmp1.detach() * torch.log(tmp2))
            if sample_weights is not None:
                instance_logits += crosOut * sample_weights[i]
            else:
                instance_logits += crosOut
    
    instance_logits = instance_logits/logits.shape[0]

    return instance_logits
    # for i in range(logits.shape[0]):
    #     tmp1 = torch.sigmoid(logits[i, :seq_len[i]]).squeeze()
    #     tmp2 = torch.sigmoid(logits2[i, :seq_len[i]]).squeeze()
    #     loss = torch.mean(-tmp1.detach() * torch.log(tmp2))
    #     instance_logits = instance_logits + loss
    # instance_logits = instance_logits/logits.shape[0]
    # return instance_logits


def train(dataloader, model, optimizer, criterion, criterion2, device, is_topk, class_weights, online_mode):
    with torch.set_grad_enabled(True):
        model.train()
        total_epoch_loss = 0
        for i, (input, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
            input = input[:, :torch.max(seq_len), :]
            input, label = input.float().to(device), label.float().to(device)

            label = label.to(torch.int64)

            if class_weights is not None:
                class_weights = class_weights.to(device)
                sample_weights = class_weights[label]
            else:
                sample_weights = None
            
            # label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=7).float()
            # encode the label to new variable so that anything greater than or equal to 1 is 1 else 0
            
            logits, logits2 = model(input, seq_len)

            clsloss = CLAS(logits, label, seq_len, criterion, device, sample_weights, is_topk)

            if online_mode == 'Binary':
                label2 = torch.where(label >= 1, torch.tensor(1).to(device), label)
                clsloss2 = CLAS2(logits2, label2, seq_len, criterion2, device, is_topk)    
            elif online_mode == 'Multi':        
                clsloss2 = CLAS(logits2, label, seq_len, criterion, device, sample_weights, is_topk)

            croloss = CENTROPY(logits, logits2, seq_len, device, online_mode, sample_weights)

            total_loss = clsloss + clsloss2 + 5*croloss

            print(f'[INFO] INPUT: {i}, clsloss: {clsloss}, clsloss2: {clsloss2}, croloss: {croloss}, Total Loss: {total_loss}')
            total_epoch_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        average_loss = total_epoch_loss / len(dataloader)
        return average_loss