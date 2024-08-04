import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchviz import make_dot

class ResNetBlock(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 
                               hidden_size, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding)
        self.conv2 = nn.Conv2d(hidden_size, 
                               output_size,
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(output_size)

        if input_size != output_size:
            self.match_dimesions = nn.Conv2d(input_size, 
                                             output_size, 
                                             kernel_size=1)
        else:
            self.match_dimesions = None
    
    def forward(self, input_layer):
        hidden_layer = self.conv1(input_layer)
        hidden_layer = self.bn1(hidden_layer)
        hidden_layer = self.relu(hidden_layer)
        output_layer = self.conv2(hidden_layer)
        output_layer = self.bn2(output_layer)
        
        if self.match_dimesions:
            input_layer = self.match_dimesions(input_layer)

        output_layer += input_layer
        output_layer = self.relu(output_layer)
        return output_layer

class ResNet(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 dropout_ratio=0.5):
        super().__init__()
        self.conv = nn.Conv2d(input_size,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)

        self.blocks = nn.ModuleList([
            ResNetBlock(64, 64, 64),
            ResNetBlock(64, 128, 128),
            ResNetBlock(128, 256, 256),
            ResNetBlock(256, 256, 512)
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []

    def forward(self, x):
        hidden_layer = self.conv(x)
        hidden_layer = self.bn(hidden_layer)
        hidden_layer = self.relu(hidden_layer)

        for block in self.blocks:
            hidden_layer = block(hidden_layer)

        hidden_layer = self.pool(hidden_layer)
        hidden_layer = hidden_layer.view(hidden_layer.size(0), -1)
        hidden_layer = self.dropout(hidden_layer)
        output_layer = self.fc(hidden_layer)
        output = self.softmax(output_layer)
        return output
    
    def train_model(self, 
                    train_loader, 
                    test_loader, 
                    device,
                    lr=1e-2, 
                    num_epochs=100, 
                    print_every=10):
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                   'min', 
                                                   factor=0.1, 
                                                   patience=5)
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            # Training phase
            self.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = correct / total
            self.train_accuracy_history.append(train_accuracy)
            epoch_loss /= len(train_loader)
            self.loss_history.append(epoch_loss)
            scheduler.step(epoch_loss)

            # Evaluation phase
            test_accuracy = self.evaluate(test_loader, device)
            self.test_accuracy_history.append(test_accuracy)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.8f}, "
                      f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

    def evaluate(self, test_loader, device):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def visualize_performance(self):
        plt.figure(figsize=(12, 5))

        # Plot Loss History
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy History
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracy_history, label='Train Accuracy')
        plt.plot(self.test_accuracy_history, label='Test Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

def loader():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='../input', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.CIFAR10(root='../input', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    return train_loader, test_loader

if __name__ == '__main__':
    # Check if a GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(input_size=3, num_classes=10)
    model.to(device)  # Move model to the GPU
    
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render('model_visualization', format='png')

    train_loader, test_loader = loader()
    model.train_model(train_loader, test_loader, device, num_epochs=50)
    model.visualize_performance()