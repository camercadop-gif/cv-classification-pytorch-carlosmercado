import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from src.models import SimpleCNN
from src.train import train_epoch
from src.evaluate import evaluate
from src.utils import EarlyStopping, set_seed

# Configuración
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Modelo
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

early_stopping = EarlyStopping(patience=4)

# Entrenamiento
for epoch in range(20):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/20")
    print(f" Train Loss: {train_loss:.4f}")
    print(f" Val Loss:   {val_loss:.4f}")
    print(f" Val Acc:    {val_acc:.2f}%")

    early_stopping(val_loss)
    if early_stopping.stop:
        print("⛔ Early stopping activado.")
        break

# Guardar modelo
torch.save(model.state_dict(), "simple_cnn.pth")
print("Modelo guardado como simple_cnn.pth")
