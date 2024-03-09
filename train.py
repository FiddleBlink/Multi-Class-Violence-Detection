import torch


def CLAS(logits, label, seq_len, criterion, device, is_topk=True):
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

    clsloss = criterion(instance_logits, label)
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

    clsloss = criterion(logits, label)
    return clsloss


def CENTROPY(logits, logits2, seq_len, device):
    instance_logits = 0.0  # tensor([])
    for i in range(logits.shape[0]):
        tmp1 = torch.softmax(logits[i], dim=0)
        tmp2 = torch.softmax(logits2[i], dim=0)
        instance_logits += -torch.mean(tmp1.detach() * torch.log(tmp2))
    
    instance_logits = instance_logits/logits.shape[0]

    return instance_logits


def train(dataloader, model, optimizer, criterion, device, is_topk):
    with torch.set_grad_enabled(True):
        model.train()
        for i, (input, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
            input = input[:, :torch.max(seq_len), :]
            input, label = input.float().to(device), label.float().to(device)

            label = label.to(torch.int64)

            # label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=7).float()

            # encode the label to new variable so that anything greater than or equal to 1 is 1 else 0
            label2 = torch.where(label >= 1, torch.tensor(1).to(device), label)
            # print("\n Label1===> ", label)
            # print("\n Label2===> ", label2)

            # print('\n',input.shape)
            # print(label.shape,'\n')

            logits, logits2 = model(input, seq_len)

            clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
            clsloss2 = CLAS2(logits2, label2, seq_len, criterion, device, is_topk)
            croloss = CENTROPY(logits, logits2, seq_len, device)

            total_loss = clsloss + clsloss2 + 5*croloss

            # print('\n',clsloss)
            # print(clsloss2)
            print(f'Epoch: {i}, Loss: {total_loss}')

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()