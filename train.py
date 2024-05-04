import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Character encoding and decoding dictionaries
def create_char_mappings(text):
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    return char_to_idx, idx_to_char

# Load data
book_fname = "./data/goblet_book.txt"
with open(book_fname, 'r') as book:
    book_data = book.read()

char_to_idx, idx_to_char = create_char_mappings(book_data)
vocab_size = len(char_to_idx)

# Model parameters
input_size = vocab_size
hidden_size = 100  # Dimensionality of the hidden state
seq_length = 25    # Length of input sequences used during training

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Learnable parameters
        self.Wf = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.Wi = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.Wo = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.Wc = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.bi = nn.Parameter(torch.zeros(hidden_size))
        self.bo = nn.Parameter(torch.zeros(hidden_size))
        self.bc = nn.Parameter(torch.zeros(hidden_size))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_states=None):
        batch_size, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((h_t, x_t), dim=1)

            f_t = torch.sigmoid(self.Wf @ combined.T + self.bf[:, None])
            i_t = torch.sigmoid(self.Wi @ combined.T + self.bi[:, None])
            o_t = torch.sigmoid(self.Wo @ combined.T + self.bo[:, None])
            c_hat_t = torch.tanh(self.Wc @ combined.T + self.bc[:, None])

            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.T)

        hidden_seq = torch.cat(hidden_seq, dim=1)
        hidden_seq = hidden_seq.reshape(batch_size, seq_len, self.hidden_size)
        output = self.fc(hidden_seq)
        return output, (h_t, c_t)

def encode_input(text):
    idxs = [char_to_idx[c] for c in text]
    return torch.nn.functional.one_hot(torch.tensor(idxs), num_classes=vocab_size).float()

def prepare_dataset(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(encode_input(data[i:i+seq_length]))
        Y.append(encode_input(data[i+1:i+seq_length+1]))
    return torch.stack(X), torch.stack(Y)

X_train, Y_train = prepare_dataset(book_data, seq_length)

model = CustomLSTM(input_size, hidden_size, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []

# Training loop
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, X_train.size(0)):
        optimizer.zero_grad()
        outputs, _ = model(X_train[i:i+1])
        loss = criterion(outputs.squeeze(), torch.max(Y_train[i], dim=2)[1])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {epoch_loss / X_train.size(0)}')

# Visualization
plt.figure()
plt.plot(losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
