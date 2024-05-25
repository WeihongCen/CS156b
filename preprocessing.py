import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import os


# Create a new directory if it does not exist
def create_folder(folder):
    if not os.path.exists(f'{folder}'):
        os.makedirs(folder)
        print(f'New directory {folder} is created!')
    return folder


def main():
    class_folder = '/groups/CS156b'
    preprocess_folder = f'{class_folder}/2024/Edgemax/preprocess'
    train_file_path = f'{class_folder}/data'
    train_csv_path = f'{train_file_path}/student_labels/train.csv'
    csv_train = pd.read_csv(train_csv_path, sep=',')

    pathologies = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                'Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other',
                'Fracture','Support Devices']
    orientations = ['Frontal', 'Lateral']

    train_data = [[[] for _ in range(len(orientations))] for _ in range(len(pathologies))]
    train_labels = [[[] for _ in range(len(orientations))] for _ in range(len(pathologies))]
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    for index, row in csv_train.iterrows():
        if index % max(1, len(csv_train) // 1000) == 0:
            print(f'{round(index / csv_train.index[-1] * 100, 2)}% completed')
        image_path = f'{train_file_path}/{row['Path']}'
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image = image.convert('RGB')
            image_tensor = transform(image)
            for i, pathology in enumerate(pathologies):
                label_tensor = torch.tensor([row[pathology]])
                if (not np.isnan(row[pathology])):
                    j = (row['Frontal/Lateral'] != 'Frontal')
                    train_data[i][j].append(image_tensor)
                    train_labels[i][j].append(label_tensor)


    for pathology in pathologies:
        for orientation in orientations:
            print(f'Train data for: {pathology} {orientation}')
            sub_folder =  f'{preprocess_folder}/{pathology.replace(' ', '_')}/{orientation}'
            create_folder(sub_folder)
            print(f'Saving {len(train_data[i][j])} training images.')
            torch.save(train_data[i][j], f'{sub_folder}/train_data_{pathology.replace(' ', '_')}_{orientation}.pt')
            print(f'Saving {len(train_labels[i][j])} training labels.')
            torch.save(train_labels[i][j], f'{sub_folder}/train_labels_{pathology.replace(' ', '_')}_{orientation}.pt')


    print('Preprocessing completed.')


if __name__ == '__main__':
    main()