import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50

class XRayDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    

def main():
    pathology = 'Enlarged Cardiomediastinum'
    orientation = 'Frontal'

    class_folder = '/groups/CS156b'
    preprocess_folder = f'{class_folder}/2024/Edgemax/preprocess'
    sub_folder =  f'{preprocess_folder}/{pathology.replace(' ', '_')}/{orientation}'
    train_data_path = f'{sub_folder}/train_data_{pathology.replace(' ', '_')}_{orientation}.pt'
    train_labels_path = f'{sub_folder}/train_labels_{pathology.replace(' ', '_')}_{orientation}.pt'
    model_path = f'{sub_folder}/model_{pathology.replace(' ', '_')}_{orientation}.pt'
    
    train_proportion = 0.8
    batch_size = 64
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device type: {device}")


    data = torch.load(train_data_path)
    labels = torch.load(train_labels_path)
    train_data = data[:int(train_proportion*len(data))]
    train_labels = labels[:int(train_proportion*len(labels))]
    val_data = data[int(train_proportion*len(data)):]
    val_labels = labels[int(train_proportion*len(labels)):]

    train_dataset = XRayDataset(train_data, train_labels)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = XRayDataset(val_data, val_labels)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


    model = resnet50(weights='ResNet50_Weights.DEFAULT')
    model.fc = nn.Sequential(nn.Linear(2048, 256),
                            nn.ReLU(), 
                            nn.Linear(256, 1),
                            nn.Tanh())
    model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    for epoch in range(num_epochs):
        idx = 0
        for inputs, labels in train_dataloader:
            labels = labels.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            if idx % max(1, len(train_dataloader) // 50) == 0:
                print(".", end ="")
            idx += 1
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    print(f'Finished Training')
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()