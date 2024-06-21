from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pandas as pd

from PIL import Image
import os

expw_lbl_to_exp = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

class expW(Dataset):
    def __init__(self, data_path, labels_path):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        column_names = ['img_name', 'face_id', 'bbox_top', 'bbox_left', 'bbox_right', 'bbox_bottom', 'bbox_confidence', 'exp_label']
        self.df = pd.read_csv(labels_path, sep=' ', names=column_names)
        
    def __len__(self):
#         return len(self.df)
        return 10000
    
    def __getitem__(self, index):
        image_data = self.df.iloc[index]
        image_path = os.path.join(self.data_path, image_data['img_name'])
        top = image_data['bbox_top']
        left = image_data['bbox_left']
        right = image_data['bbox_right']
        bottom = image_data['bbox_bottom']
        label = image_data['exp_label']
        
        image = Image.open(image_path)
        cropped_image = TF.crop(image, top, left, bottom - top, right - left)
        cropped_image = self.transform(cropped_image)
        
        return cropped_image, label