import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SEMData(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, f) 
                            for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # all images are 1024 x 768, no resize
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image = self.transform(image)
        return image

set = SEMData("dataset/Patterned_surface")
print(set.__len__())

