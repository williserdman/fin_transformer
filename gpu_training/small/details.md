SEQ_LEN = 128
BATCH_SIZE = 128

train_dataset = MyDataset(all_sequences[:int(T*0.8)], all_targets[:int(T*0.8)], SEQ_LEN)
val_dataset = MyDataset(all_sequences[int(T*0.8):], all_targets[int(T*0.8):], SEQ_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# pass frequencies to the model (as a list)

model = SimpleTransformer(vocab, SEQ_LEN, number_of_classes, output_class_freq=class_freqs, attention_layers=10, hidden_dim=128)

res = train_loop(model, train_dataloader, val_dataloader, 1000, 4e-6, 1e-5, torch.device("cuda"), "checkpoint", 50)
