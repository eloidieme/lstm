import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import time

book_fname = "./data/goblet_book.txt"
with open(book_fname, 'r') as book:
    book_data = book.read()
len(book_data)

word_list = book_data.split()
chars = [[*word] for word in word_list]
max_len = max(len(word) for word in chars)
for wordl in chars:
    while len(wordl) < max_len:
        wordl.append(' ')
chars = np.array(chars)

unique_chars = list(np.unique(chars))
unique_chars.append('\n')
unique_chars.append('\t')
K = len(unique_chars)  # dimensionality of the input and output vectors

char_to_ind = {}
ind_to_char = {}
for idx, char in enumerate(unique_chars):
    char_to_ind[char] = idx
    ind_to_char[idx] = char

m = 100  # dimensionality of the hidden state
eta = 0.1  # learning rate
seq_length = 25  # length of input sequences used during training
epsilon = 1e-8  # for AdaGrad

sig = 0.01
Wf = torch.normal(0.0, sig, (m, K), dtype=torch.double, requires_grad=True)
Wi = torch.normal(0.0, sig, (m, K), dtype=torch.double, requires_grad=True)
Wo = torch.normal(0.0, sig, (m, K), dtype=torch.double, requires_grad=True)
Wc = torch.normal(0.0, sig, (m, K), dtype=torch.double, requires_grad=True)
Wlist = [Wf, Wi, Wo, Wc]
Wall = torch.cat(Wlist, dim=0)

Uf = torch.normal(0.0, sig, (m, m), dtype=torch.double, requires_grad=True)
Ui = torch.normal(0.0, sig, (m, m), dtype=torch.double, requires_grad=True)
Uo = torch.normal(0.0, sig, (m, m), dtype=torch.double, requires_grad=True)
Uc = torch.normal(0.0, sig, (m, m), dtype=torch.double, requires_grad=True)
Ulist = [Uf, Ui, Uo, Uc]
Uall = torch.cat(Ulist, dim=0)

V = torch.normal(0.0, sig, (K, m), dtype=torch.double, requires_grad=True)
c = torch.zeros((K, 1), dtype=torch.double, requires_grad=True)
FClist = [V, c]
FC = torch.cat(FClist, dim=1)

E1 = torch.cat([torch.eye(m), torch.zeros((m, m)), torch.zeros((m, m)), torch.zeros((m, m))], dim=1)
E2 = torch.cat([torch.zeros((m, m)), torch.eye(m), torch.zeros((m, m)), torch.zeros((m, m))], dim=1)
E3 = torch.cat([torch.zeros((m, m)), torch.zeros((m, m)), torch.eye(m), torch.zeros((m, m))], dim=1)
E4 = torch.cat([torch.zeros((m, m)), torch.zeros((m, m)), torch.zeros((m, m)), torch.eye(m)], dim=1)

LSTM = {
    'Wall': Wall, 
    'Uall': Uall,
    'FC': FC
}

def encode_char(char):
    oh = [0]*K
    oh[char_to_ind[char]] = 1
    return oh

def synthetize_seq(lstm, h0, c0, x0, n, T = 1):
    t, ht, ct, xt = 0, h0.clone(), c0.clone(), x0.clone().reshape((K, 1))
    indexes = []
    while t < n:
        at = torch.mm(lstm['Wall'], ht) + torch.mm(lstm['Uall'], xt)
        ft = F.sigmoid(torch.mm(E1, at))
        it = F.sigmoid(torch.mm(E2, at))
        ot = F.sigmoid(torch.mm(E3, at))
        ctilde = F.tanh(torch.mm(E4, at))
        ct = ft * ct + it * ctilde
        ht = ot * F.tanh(ct)
        out = torch.mm(lstm['FC'][:, :-1], ht) + lstm['FC'][:, -1:]
        pt = F.softmax(out/T, dim=0)
        cp = torch.cumsum(pt, dim=0)
        a = torch.rand(1)
        ixs = torch.where(cp - a > 0)
        ii = ixs[0][0].item()
        indexes.append(ii)
        xt = torch.zeros((K, 1), dtype=torch.double)
        xt[ii, 0] = 1
        t += 1
    Y = []
    for idx in indexes:
        oh = [0]*K
        oh[idx] = 1
        Y.append(oh)
    Y = torch.tensor(Y).t()
    
    s = ''
    for i in range(Y.shape[1]):
        idx = torch.where(Y[:, i] == 1)[0].item()
        s += ind_to_char[idx]
    
    return Y, s

def encode_string(chars):
    M = []
    for i in range(len(chars)):
        M.append(encode_char(chars[i]))
    M = torch.tensor(M, dtype=torch.double).t()
    return M

def forward(lstm, X, hprev, cprev):
    ht = hprev.clone()
    ct = cprev.clone()
    P = torch.zeros((K, seq_length), dtype=torch.double)
    for i in range(seq_length):
        xt = X[:, i].reshape((K, 1))
        at = torch.mm(lstm['Wall'], ht) + torch.mm(lstm['Uall'], xt)
        ft = F.sigmoid(torch.mm(E1, at))
        it = F.sigmoid(torch.mm(E2, at))
        ot = F.sigmoid(torch.mm(E3, at))
        ctilde = F.tanh(torch.mm(E4, at))
        ct = ft * ct + it * ctilde
        ht = ot * F.tanh(ct)
        out = torch.mm(lstm['FC'][:, :-1], ht) + lstm['FC'][:, -1:]
        pt = F.softmax(out, dim=0)

        P[:, i] = pt.squeeze()

    return P, ht, ct

def compute_loss(Y, P):
    log_probs = torch.log(P)
    cross_entropy = -torch.sum(Y * log_probs)
    loss = cross_entropy.item()
    return loss
############################# TRAINING - AdaGrad ############################# 

e, step, epoch = 0, 0, 0
n_epochs = 2
smooth_loss = 0
seq_length = 25
losses = []
hprev = torch.zeros((m, 1), dtype=torch.double)
cprev = torch.zeros((m, 1), dtype=torch.double)

mWf = torch.zeros_like(Wf, dtype=torch.double)
mWi = torch.zeros_like(Wi, dtype=torch.double)
mWo = torch.zeros_like(Wo, dtype=torch.double)
mWc = torch.zeros_like(Wc, dtype=torch.double)
mUf = torch.zeros_like(Uf, dtype=torch.double)
mUi = torch.zeros_like(Ui, dtype=torch.double)
mUo = torch.zeros_like(Uo, dtype=torch.double)
mUc = torch.zeros_like(Uc, dtype=torch.double)
mV = torch.zeros_like(V, dtype=torch.double)
mc = torch.zeros_like(c, dtype=torch.double)
msW = {
    'Wf': mWf, 
    'Wi': mWi, 
    'Wo': mWo, 
    'Wc': mWc, 
}
msU = {
    'Uf': mUf,
    'Ui': mUi,
    'Uo': mUo,
    'Uc': mUc
}
msFC = {
    'V': mV,
    'c': mc
}
Ws = ['Wf', 'Wi', 'Wo', 'Wc']
Us = ['Uf', 'Ui', 'Uo', 'Uc']
FCs = ['V', 'c']

while epoch < n_epochs:
    for p in Wlist + Ulist + FClist:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    X_chars = book_data[e:e+seq_length]
    Y_chars = book_data[e+1:e+seq_length+1]
    X_train = encode_string(X_chars)
    Y_train = encode_string(Y_chars)

    P_train, ht, ct = forward(LSTM, X_train, hprev, cprev)
    cross_entropy, loss = compute_loss(Y_train, P_train)
    cross_entropy.backward()

    for idx, key in enumerate(Ws):
        grad = torch.clamp(Wlist[idx].grad, -5, 5)
        msW[key] += grad**2
        LSTM['Wall'][100*idx:100*(idx+1)] -= (eta/torch.sqrt(msW[key] + epsilon))*grad

    for idx, key in enumerate(Us):
        grad = torch.clamp(Ulist[idx].grad, -5, 5)
        msU[key] += grad**2
        LSTM['Uall'][100*idx:100*(idx+1)] -= (eta/torch.sqrt(msU[key] + epsilon))*grad

    for idx, key in enumerate(FCs):
        grad = torch.clamp(FClist[idx].grad, -5, 5)
        msFC[key] += grad**2
        LSTM['FC'][:, -1*idx:-1*(1-idx)] -= (eta/torch.sqrt(msFC[key] + epsilon))*grad

    if step == 0:
        smooth_loss = loss
    else:
        smooth_loss = 0.999*smooth_loss + 0.001*loss

    losses.append(smooth_loss)

    if step % 1000 == 0:
        print(f"Step: {step}")
        print(f"\t * Smooth loss: {smooth_loss}")
    if step % 5000 == 0:
        _, s_syn = synthetize_seq(LSTM, hprev, cprev, X_train[:, 0], 200, 0.6)
        print("-" * 100)
        print(f"Synthetized sequence: \n{s_syn}")
        print("-" * 100)
    if step % 100000 == 0 and step > 0:
        _, s_lsyn = synthetize_seq(LSTM, hprev, cprev, X_train[:, 0], 1000, 0.6)
        print("-" * 100)
        print(f"Long synthetized sequence: \n{s_lsyn}")
        print("-" * 100)

    step += 1
    e += seq_length
    if e > len(book_data) - seq_length:
        e = 0
        epoch += 1
        hprev = torch.zeros((m, 1), dtype=torch.double)
        cprev = torch.zeros((m, 1), dtype=torch.double)
    else:
        hprev = ht.detach()
        cprev = ct.detach()

with open(f'rnn_{time.time()}.pickle', 'wb') as handle:
    pickle.dump(LSTM, handle, protocol=pickle.HIGHEST_PROTOCOL)
##############################################################################

plt.grid(True)
plt.plot(losses)
plt.xlabel('Steps')
plt.ylabel('Smooth loss')
plt.title(f'Training with AdaGrad - eta: {eta} - seq_length: {seq_length} - m: {m} - n_epochs: {n_epochs}')
plt.savefig('./training.png')

with open('rnn.pickle', 'rb') as handle:
    test_rnn = pickle.load(handle)

first_char = " "
x_input = encode_string(first_char)
Y_t, s_t = synthetize_seq(
    test_rnn, 
    torch.zeros((m, 1), dtype=torch.double), 
    x_input[:,0], 1000, 0.8)
print(first_char + s_t)