import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

pathologies = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                'Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other',
                'Fracture','Support Devices']
orientations = ['Frontal', 'Lateral']


def create_folder(folder):
    if not os.path.exists(f'{folder}'):
        os.makedirs(folder)
        print(f"New directory {folder} is created!")
    return folder

def main():
    test_csv_path = "/groups/CS156b/data/student_labels/test_ids.csv"
    csv_test = pd.read_csv(test_csv_path, sep=',')

    class_folder = '/groups/CS156b'
    test_file_path = f'{class_folder}/data/'
    group_folder = f'{class_folder}/2024/Edgemax'
    test_folder = f'{group_folder}/test'
    create_folder(test_folder)

    transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 0 = Frontal images, 1 = Lateral images
    test_data = [[], []]
    for index, row in csv_test.iterrows():
        if index % max(1, len(csv_test) // 1000) == 0:
            print(f"{round(index / csv_test.index[-1] * 100, 2)}% completed")
        if os.path.exists(test_file_path + row['Path']):
            image = Image.open(test_file_path + row['Path'])
            image = image.convert('RGB')
            image_tensor = transform(image)
            i = ('frontal' not in row['Path'])
            test_data[i].append(image_tensor)

    for i, orientation in enumerate(orientations):
        print(f'Saving {len(test_data[i])} images of orientation {orientation}')
        test_processed_imgs_path = f'{test_folder}/test_{orientation}.pt'
        torch.save(test_data[i], test_processed_imgs_path)
    
    print('Test preprocessing completed.')


if __name__ == '__main__':
    main()