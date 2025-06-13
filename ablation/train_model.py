import os
import fire
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datetime import datetime
from utils.logger import CustomLogger
from load_datasets.cifar100 import load_cifar100_dataloaders
from utils.misc import load_model_and_targets, GetDevice, adjust_model_for_num_classes
logger = CustomLogger(__name__).logger

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

def train_cifar_model(model_name: str = 'resnet',
                      data_root: str = './data',
                      output_path: str = './.weights/model.pth',
                      batch_size: int = 1024,
                      epochs: int = 30,
                      lr: float = 1e-5,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      seed: int = 42):

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load data
    train_loader, val_loader = load_cifar100_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        include_train=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Load model
    device = GetDevice()
    model, _ = load_model_and_targets(model_name, device=device)
    model = adjust_model_for_num_classes(model, 100)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_acc = 0.0
    best_model = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate_accuracy(model, val_loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            logger.info(f"New best model at epoch {epoch+1} with validation accuracy {val_acc:.2f}%")

    # Tag model with timestamp and seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_best.pth"
    final_path = os.path.join(os.path.dirname(output_path), filename)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(best_model, final_path)
    logger.info(f"Best model saved to {final_path} with accuracy {best_acc:.2f}%")


if __name__ == '__main__':
    fire.Fire(train_cifar_model)
