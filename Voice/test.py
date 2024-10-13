import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchaudio.transforms import PitchShift
import matplotlib.pyplot as plt


# Tokenizer
def tokenize(text):
    return [vocab.get(char, vocab["<unk>"]) for char in text.lower()]  # char -> index


# Build vocabulary
vocab = {"<pad>": 0, "<unk>": 1}  # Add <pad> and <unk>
for i, char in enumerate("abcdefghijklmnopqrstuvwxyz ", 2):
    vocab[char] = i


# Dataset class for Common Voice with TSV
class CommonVoiceDataset(Dataset):
    def __init__(self, tsv_file, audio_folder):
        # Đọc file TSV
        self.data = pd.read_csv(tsv_file, sep='\t')  # change separator
        self.audio_folder = audio_folder
        self.mel_transform = MelSpectrogram(sample_rate=16000, n_mels=80)

    def process_text(self, text):
        return text  # let's assume that text is already tokenized

    def process_audio(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

        # Data augmentation: add noise and pitch shift
        noise = torch.randn_like(waveform) * 0.005
        augmented_waveform = waveform + noise
        augmented_waveform = torchaudio.transforms.PitchShift(sample_rate=8000, n_steps=2)(augmented_waveform)

        mel_spec = self.mel_transform(augmented_waveform)
        return mel_spec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.process_text(self.data.iloc[idx]['sentence'])  # colum 'sentence' in TSV
        audio_path = os.path.join(self.audio_folder, self.data.iloc[idx]['path'])  # colum 'path' have audio path
        audio = self.process_audio(audio_path)
        return text, audio


class TTSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, output_dim, num_lstm_layers, dropout):
        super(TTSModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True,
                            dropout=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_dim, num_heads=8, device='cuda')
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_input):
        embedded = self.embedding(text_input)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, lstm_hidden_dim)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = self.fc(self.dropout(attn_output))  # (batch_size, seq_len, output_dim)
        return output


def collate_fn(batch):
    text_data, audio_data = zip(*batch)
    tokenized_texts = [torch.tensor(tokenize(text), dtype=torch.long) for text in text_data]
    padded_texts = pad_sequence(tokenized_texts, batch_first=True, padding_value=vocab["<pad>"])
    max_length = max([audio.size(2) for audio in audio_data])
    padded_audio = [torch.nn.functional.pad(audio, (0, max_length - audio.size(2))) for audio in audio_data]
    padded_audio = torch.stack(padded_audio)
    return padded_texts, padded_audio


def train(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_values = []
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # Expecting (batch_size, seq_len, output_dim)
            outputs = outputs.permute(1, 0, 2)  # CTC expects (seq_len, batch_size, num_classes)
            log_probs = F.log_softmax(outputs, dim=2)

            # Create input_lengths (sequence lengths for each batch)
            input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long).to(
                device)

            # Flatten targets and calculate target_lengths
            target_lengths = torch.tensor([len(t[t != vocab["<pad>"]]) for t in targets], dtype=torch.long).to(device)
            targets = targets[targets != vocab["<pad>"]].view(-1).to(device)  # Flatten targets

            # Calculate the loss using CTCLoss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_values.append(epoch_loss)
        scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    # Plot loss after training
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_values, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()


tsv_file = 'cv-corpus-19.0-2024-09-13/vi/train.tsv'
audio_folder = 'cv-corpus-19.0-2024-09-13/vi/clips/'

vocab_size = len(vocab)
embedding_dim = 1024
lstm_hidden_dim = 2048
output_dim = 80
num_lstm_layers = 6
dropout = 0.5
epochs = 50

dataset = CommonVoiceDataset(tsv_file, audio_folder)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

model = TTSModel(vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 lstm_hidden_dim=lstm_hidden_dim,
                 output_dim=output_dim,
                 num_lstm_layers=num_lstm_layers,
                 dropout=dropout)

criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

train(model, dataloader, criterion, optimizer, scheduler, epochs)