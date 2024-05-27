import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from collections import OrderedDict
import csv
import os

pathologies = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                'Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other',
                'Fracture','Support Devices']
orientations = ['Frontal', 'Lateral']

class_folder = '/groups/CS156b'
group_folder = f'{class_folder}/2024/Edgemax'
test_folder = f'{group_folder}/test'
model_path = "/groups/CS156b/2024/Edgemax/model"
n_epochs = 20

def validate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device type: {device}")
    labels_dict = {}
    
    for orientation in orientations:
        test_processed_imgs_path = f'{test_folder}/test_{orientation}_imgs.pt'
        test_processed_ids_path = f'{test_folder}/test_{orientation}_ids.pt'
        test_orientation_data = torch.load(test_processed_imgs_path)
        test_orientation_ids = torch.load(test_processed_ids_path)
        orientation_models = []

        for i, pathology in enumerate(pathologies):
            model = resnet50(weights='ResNet50_Weights.DEFAULT')
            model.fc = nn.Sequential(nn.Linear(2048, 256),
                                    nn.ReLU(), 
                                    nn.Linear(256, 1),
                                    nn.Tanh())
            model = nn.DataParallel(model)
            model.to(device)
            orientation_models.append(model)
            pathology_model_path = f'{model_path}/{pathology.replace(' ', '_')}/{orientation}/model_{pathology.replace(' ', '_')}_{orientation}.pt'
            orientation_models[i].load_state_dict(torch.load(pathology_model_path))

        test_dataloader = DataLoader(test_orientation_data, batch_size=1, shuffle=False)

        for epoch in range(n_epochs):
            print(f'Epoch {epoch+1}/{n_epochs}:', end='')
            for idx, images in enumerate(test_dataloader):
                labels_dict[test_orientation_ids[idx]] = [test_orientation_ids[idx]]
                for pathology_model in orientation_models:
                    pathology_model.eval()
                    images = images.to(device)
                    outputs = pathology_model(images)
                    test_predicted = (torch.squeeze(outputs.data)).float().item()
                    labels_dict[idx].append(test_predicted)

        print(f'Finished Testing {orientation}.')
    write_to_csv(labels_dict)

def write_to_csv(labels_dict):
    labels_dict_sorted = OrderedDict(sorted(labels_dict.items()))
    header = ['Id', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
              'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    with open('test_submission.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for key in labels_dict_sorted:
            csvwriter.writerow(labels_dict_sorted[key])
    print("Writing results to test_submission.csv finished.")

if __name__ == '__main__':
    validate()


