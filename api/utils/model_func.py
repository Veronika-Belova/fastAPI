import requests
import json
import torch
from torchvision.models import resnet18
import torchvision.transforms as T

#Загрузка imagenet_class_index

# url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
# response = requests.get(url)
# if response.status_code == 200:
#     data = response.json()
#     with open('imagenet_class_index.json', 'w') as file:
#         json.dump(data, file)
#     print("Данные успешно сохранены в файл 'imagenet_class_index.json'")
# else:
#     print("Не удалось получить данные по указанному URL")


def load_classes():
    with open('utils/imagenet_class_index.json') as f:
        labels = json.load(f)
    return labels

def class_id_to_label(i):
    labels = load_classes()
    class_info = labels.get(str(i))
    if class_info:
        return class_info[1]
    else:
        return "Class not found"
    
def load_model():
    model = resnet18()
    model.load_state_dict(torch.load('utils/resnet18-weights.pth', map_location='cpu'))
    model.eval()
    return model

def transform_image(img):
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return transform(img)