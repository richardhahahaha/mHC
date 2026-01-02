import math
from pathlib import Path
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mhyperconn import HyperConnectionDecodeTransformer


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, self.data.numel() - (self.block_size + 1))

    def __getitem__(self, idx: int):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        if ids.dim() != 1:
            ids = ids.view(-1)
        return "".join(self.itos[int(i)] for i in ids)


@torch.no_grad()
def generate(model: nn.Module, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    model.eval()
    ids = prompt_ids
    device = next(model.parameters()).device
    ids = ids.to(device)
    for _ in range(max_new_tokens):
        if hasattr(model, "max_len") and ids.numel() > model.max_len:
            ctx = ids[-model.max_len:]
        else:
            ctx = ids
        logits = model(ctx.unsqueeze(0))[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id.squeeze(0)], dim=0)
    return ids


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 5
    lr = 3e-4
    block_size = 128

    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "tinyshakespeare.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not data_path.exists():
        print(f"Downloading tinyshakespeare to {data_path} ...")
        try:
            urllib.request.urlretrieve(url, data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from {url}: {e}")

    text = data_path.read_text(encoding="utf-8")

    ds = CharDataset(text=text, block_size=block_size)
    n = len(ds)
    split = int(n * 0.9)
    train_ds = torch.utils.data.Subset(ds, range(0, split))
    test_ds = torch.utils.data.Subset(ds, range(split, n))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
 
    model = HyperConnectionDecodeTransformer(
        vocab_size=ds.vocab_size,
        max_len=block_size,
        dim=128,
        n_layers=6,
        n_heads=4,
        rate=4,
        dropout=0.1,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {param_count}")
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    print("Start training ...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_tokens = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
 
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, ds.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
 
            train_loss_sum += loss.item() * y.numel()
            train_tokens += y.numel()
 
        model.eval()
        test_loss_sum = 0.0
        test_tokens = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits.reshape(-1, ds.vocab_size), y.reshape(-1))
                test_loss_sum += loss.item() * y.numel()
                test_tokens += y.numel()
 
        train_loss = train_loss_sum / max(train_tokens, 1)
        test_loss = test_loss_sum / max(test_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20.0))
        test_ppl = math.exp(min(test_loss, 20.0))
 
        prompt = "mHC: "
        prompt_ids = ds.encode(prompt)
        out_ids = generate(model, prompt_ids, max_new_tokens=120, temperature=0.9)
        out_text = ds.decode(out_ids)
 
        print(
            f"epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} ppl {train_ppl:.2f} | "
            f"test loss {test_loss:.4f} ppl {test_ppl:.2f}"
        )
        print(out_text)
 
 
if __name__ == "__main__":
    main()
