import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import make_model, make_std_mask

warnings.filterwarnings(action="ignore")


def train_epoch(model, train_data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for i, (src, tgt) in enumerate(train_data_loader):
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        src_mask, tgt_mask = make_std_mask(src, tgt_input, pad=0)
        out = model.forward(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(
            out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)
        )
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 20 == 0:
            print(f"  Step {i}, Loss: {loss.item():.4f}")

    return total_loss / len(train_data_loader)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = 100
    N_SAMPLES = 2000
    SEQ_LEN = 10
    data = torch.randint(1, V, (N_SAMPLES, SEQ_LEN))
    dataset = TensorDataset(data, data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = make_model(V, V, N=2, d_model=128, d_ff=256, h=4).to(DEVICE)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    e = int(input(":"))
    for epoch in range(e):
        print(f"Epoch {epoch}")
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"epoch {epoch} | avg loss: {avg_loss:.4f}")
