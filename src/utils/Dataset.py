from torch.utils.data import Dataset
from PIL import Image
import os

class ImageSet(Dataset):
    def __init__(self, photo_path, monet_path, transform):
        self.photo_path = photo_path
        self.monet_path = monet_path
        self.transform = transform
        self.photos = os.listdir(photo_path)
        self.monets = os.listdir(monet_path)
        self.len_photo = len(self.photos)
        self.len_monet = len(self.monets)

    def __len__(self):
        return max(self.len_photo, self.len_monet)

    def __getitem__(self, index):
        photo = Image.open(self.photo_path + self.photos[index % self.len_photo]).convert("RGB")
        monet = Image.open(self.monet_path + self.monets[index % self.len_monet]).convert("RGB")

        photo = self.transform(photo)
        monet = self.transform(monet)

        return photo, monet

