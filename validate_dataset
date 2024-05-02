import torch
from main import FlowerCNN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 32
valid_dataset = datasets.ImageFolder(r'flower_test_dataset', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)



def main():
    model = FlowerCNN()

    model.load_state_dict(torch.load('res_weights/flower_cnn_base.pth'), map_location=torch.device('cpu'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    main()
