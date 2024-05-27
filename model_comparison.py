import os
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
    

def create_folder(folder):
    if not os.path.exists(f'{folder}'):
        os.makedirs(folder)
        print(f"New directory {folder} is created!")
    return folder


def train(pathology, orientation, batch_size, learning_rate, momentum, num_epochs):
    print(f'Training began for {pathology} {orientation}')

    class_folder = '/groups/CS156b'
    preprocess_folder = f'{class_folder}/2024/Edgemax/preprocess'
    preprocess_sub_folder =  f'{preprocess_folder}/{pathology.replace(' ', '_')}/{orientation}'
    train_data_path = f'{preprocess_sub_folder}/train_data_{pathology.replace(' ', '_')}_{orientation}.pt'
    train_labels_path = f'{preprocess_sub_folder}/train_labels_{pathology.replace(' ', '_')}_{orientation}.pt'
    model_folder = f'{class_folder}/2024/Edgemax/model_comparison'
    model_sub_folder = f'{model_folder}/{pathology.replace(' ', '_')}/{orientation}'
    model_path = f'{model_sub_folder}/model_{pathology.replace(' ', '_')}_{orientation}.pt'
    create_folder(model_sub_folder)
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device type: {device}')


    train_proportion = 0.8
    print('Loading training data')
    data = torch.load(train_data_path)
    labels = torch.load(train_labels_path)
    print(f'Loaded {len(data)} training images')
    train_data = data[:int(train_proportion*len(data))]
    train_labels = labels[:int(train_proportion*len(labels))]
    val_data = data[int(train_proportion*len(data)):]
    val_labels = labels[int(train_proportion*len(labels)):]

    train_dataset = XRayDataset(train_data, train_labels)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = XRayDataset(val_data, val_labels)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    print('Initializing pre-trained model')
    model = resnet50(weights='ResNet50_Weights.DEFAULT')
    model.fc = nn.Sequential(nn.Linear(2048, 256),
                            nn.ReLU(), 
                            nn.Linear(256, 1),
                            nn.Tanh())
    model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    best_val_loss = 4
    best_epoch = 0
    val_loss_history = [best_val_loss]
    base_patience = 2
    patience = base_patience # Stop training when the model hasn't improved for n epochs
    print('Begin training')
    for epoch in range(num_epochs):
        model.train()
        for idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            if idx % max(1, len(train_dataloader) // 10) == 0:
                print('.', end ='')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}', end ='')
    
        val_loss = 0
        with torch.no_grad():
            model.eval()
            for idx, (images, labels) in enumerate(val_dataloader):
                if idx % max(1, len(val_dataloader) // 10) == 0:
                    print('.', end ='')
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)
        val_loss_history.append(val_loss)
        print(f'Validation loss: {val_loss}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
        
        if val_loss >= val_loss_history[epoch]:
            patience -= 1
        else:
            patience = base_patience

        if patience == 0:
            print(f'Chosen model at epoch {best_epoch+1}')
            break

    print(f'Finished Training for {pathology} {orientation} with validation loss: {best_val_loss}')
    

def main():
    pathology = 'Enlarged Cardiomediastinum'
    orientation = 'Frontal'

    batch_size = [32, 64, 128, 256]
    learning_rate = [0.001, 0.003, 0.01]
    momentum = [0.8, 0.9, 0.99]
    num_epochs = 10

    
    train(pathology, orientation, batch_size, learning_rate, momentum, num_epochs)
    print('')

    print(f'Finished Training all models')


if __name__ == '__main__':
    main()