import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from typing import Literal, Optional, Callable
from config import DATA_DIR
from utils import read_json_file

class CaptionDataset(Dataset):
    """A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches."""

    def __init__(
            self, 
            data_name: str = "caps3_freq5_maxlen100", 
            split: Literal['train', 'val', 'test'] = 'train', 
            transform: Optional[Callable] = None):
        
        self.split = split
        self.h5py_path = os.path.join(DATA_DIR, "h5py", self.split + '_' +  data_name + '.hdf5')
        self.captions_path = os.path.join(DATA_DIR, "captions", self.split + '_' + data_name + '.json')

        self.h5py_file = h5py.File(self.h5py_path, 'r')
        self.imgs_data = self.h5py_file['images']
        self.caps_per_img = self.h5py_file.attrs['captions_per_image']
        self.captions_data = read_json_file(self.captions_path)
        self.transform = transform
        self.dataset_size = len(self.captions_data) * self.caps_per_img

    def __getitem__(self, i):
        id = i // self.caps_per_img
        cap_id = i % self.caps_per_img
        img_data = self.imgs_data[id] # tuple(img_path, img_value)
        caption_data = self.captions_data[id]

        assert img_data[0].decode('utf-8') == caption_data["img_path"]
        img_path = caption_data["img_path"]
        img = torch.FloatTensor( img_data[1] / 255.) # 
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(caption_data["enc_captions"][cap_id])
        caplen = torch.LongTensor([caption_data["caplens"][cap_id]])

        if self.split == 'train':
            return img, caption, caplen
        else:
            # For validation, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(caption_data["enc_captions"][:self.caps_per_img])
            return img, caption, caplen, all_captions, img_path

    def __len__(self):
        return self.dataset_size
    

caption_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])