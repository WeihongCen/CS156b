import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
import os

pathologies = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                'Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other',
                'Fracture','Support Devices']
orientations = ['Frontal', 'Lateral']

class_folder = '/groups/CS156b'
group_folder = f'{class_folder}/2024/Edgemax'
test_folder = f'{group_folder}/test'

def validate(test_data, models):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device type: {device}")

    model = resnet50(weights='ResNet50_Weights.DEFAULT')
    model.fc = nn.Sequential(nn.Linear(2048, 256),
                            nn.ReLU(), 
                            nn.Linear(256, 1),
                            nn.Tanh())
    model = nn.DataParallel(model)
    model.to(device)
    
    for orientation in orientations:
        test_data_path = f'{test_folder}/test_{orientation}.pt'
        test_orientation_data = torch.load(test_data_path)
        orientation_models = []
        model_path = "/groups/CS156b/2024/Edgemax/preprocess"
        for pathology in os.listdir(model_path):
            pathology_model_path = f'{model_path}/{pathology}/{orientation}'
            model.load_state_dict(torch.load(pathology_model_path))
            orientation_models.append()
        for pathology_model in orientation_models:
            pathology_model.eval()
            output = pathology_model(test_data)
            test_predicted = (torch.squeeze(output.data)).float()
    pass