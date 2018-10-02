import numpy as np
import pydicom
from torch.utils.data import Dataset
from PIL import Image
import os


class ChestXrayDataSet(Dataset):

    def __init__(
            self,
            data_dir=None,
            png_or_dcm='dcm',
            image_list_file=None,
            transform=None):
        """
        Utility for loading x-ray images in either dcm (DICOM) or png formats.
        :param path_to_images: Path to image directory
        :param transform: Transformation to apply to the image before returning
        :param sample: Nothing if None. If integer, select random sample of size sample.
        :param png_or_dcm: Specify if image format is 'dcm' or 'png'
        :param image_list_file: path to the file containing images with corresponding labels.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.png_or_dcm = png_or_dcm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        if 'dcm' in self.png_or_dcm:
            pic = pydicom.read_file(image_name)
            pic = pic.pixel_array
            max_val = np.max(pic)
            # rescale to be between 0-255
            pic_scaled = np.array(pic) / max_val * 255.
            # pic_scaled = []
            # for row in pic:
            #     row_scaled = []
            #     for col in row:
            #         col_scaled = int((float(col) / float(max_val)) * 255.0)
            #         row_scaled.append(col_scaled)
            #     pic_scaled.append(row_scaled)
            # pic_scaled = np.array(pic_scaled)
            image = Image.fromarray(pic_scaled.astype('uint8'), 'L')
            image = image.convert('RGB')
        elif 'png' in self.png_or_dcm:
            image = Image.open(image_name)
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, torch.FloatTensor(label)


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    cxr = ChestXrayDataSet(data_dir='./RSNA/images',
                           png_or_dcm='dcm',
                           image_list_file='./RSNA/labels/train_list.txt',
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.TenCrop(224),
                               transforms.Lambda
                               (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                               transforms.Lambda
                               (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                           ]))

    print(cxr[0][0].shape)
