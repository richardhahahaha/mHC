import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mhyperconn import ImageHyperConnectionTransformer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data"
    batch_size = 128
    epochs = 5
    lr = 3e-4

    image_size = 32
    patch_size = 4

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = ImageHyperConnectionTransformer(
        image_size=(image_size, image_size),
        patch_size=(patch_size, patch_size),
        in_channels=1,
        num_classes=10,
        dim=96,
        n_layers=6,
        n_heads=4,
        rate=2,
        dropout=0.1,
        pool_size=4,
        mask_ratio=0.0,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {param_count}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    print("Start training ...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                test_loss += loss.item() * targets.size(0)
                test_correct += (logits.argmax(dim=1) == targets).sum().item()
                test_total += targets.size(0)

        print(
            f"epoch {epoch}/{epochs} | "
            f"train loss {train_loss / train_total:.4f} acc {train_correct / train_total:.4f} | "
            f"test loss {test_loss / test_total:.4f} acc {test_correct / test_total:.4f}"
        )


if __name__ == "__main__":
    main()
