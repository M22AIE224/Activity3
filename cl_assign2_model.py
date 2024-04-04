import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

if(torch.cuda.is_available()):
  device = "cuda"
else:
  device = "cpu"

def plot_training_curves(train_loss_history, train_accuracy_history, optimizer_name):
    plt.plot(range(1, 10 + 1), train_loss_history, label='Training Loss')
    plt.plot(range(1, 10 + 1), train_accuracy_history, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title(f'Training Loss and Accuracy with {optimizer_name}')
    plt.legend()
    plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

model = torchvision.models.resnet101(pretrained=True)
optimizers = [
    optim.Adam(model.fc.parameters()),
    optim.Adagrad(model.fc.parameters()),
    optim.Adadelta(model.fc.parameters()),
    optim.RMSprop(model.fc.parameters())
]


criterion = nn.CrossEntropyLoss()
for optimizer in optimizers:
    train_loss_history = []
    train_accuracy_history = []

    optimizer_name = optimizer.__class__.__name__
    print(f"Training with Optimizer {optimizer_name} optimizer...")

    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100. * correct / total

        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{10}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    top_5_train_accuracy = np.sort(train_accuracy_history)[-5:]
    print("---------------------------------------------------------")
    print("Top 5 Training Accuracies:", top_5_train_accuracy)
    print("---------------------------------------------------------")
    plot_training_curves(train_loss_history, train_accuracy_history, optimizer_name)
