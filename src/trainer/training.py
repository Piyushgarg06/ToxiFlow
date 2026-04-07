def training(model, dataLoader, compute_loss, optimizer):
    model.train()
    total_loss = 0
    for batch in dataLoader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        weight = batch['weight']
        target = batch['target']
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, target, weight)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss = total_loss / len(dataLoader)
    return avg_loss
        