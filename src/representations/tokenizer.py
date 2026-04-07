def tokenize_data(data, tokenizer, max_length):
    tokenized=[]
    for item in data:
        encoded = tokenizer(
            item['input'],
            max_length=max_length,
            truncation=True, 
            padding='max_length',
            return_tensors=None
            )
        new_item = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'target': item['target'],
            'weight': item['weight'],
        }
        tokenized.append(new_item)
    return tokenized