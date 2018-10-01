import pandas as pd
import numpy as np
import pydicom
from torch.utils.data import Dataset
import os
from PIL import Image


class CXRDataset(Dataset):

    def __init__(
            self,
            path_to_images,
            transform=None,
            sample=0,
            # finding="any",
            path_to_labels=None):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv(path_to_labels, index_col='patientId')
        self.df['Target'] *= 6
        #         self.df = self.df[self.df['fold'] == fold]

        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        assert len(self.df) > sample > 0, 'Sample is either less than 0 or larger than the labels.'
        self.df = self.df.sample(sample)

        # if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
        #     if finding in self.df.columns:
        #         if len(self.df[self.df[finding] == 1]) > 0:
        #             self.df = self.df[self.df[finding] == 1]
        #         else:
        #             print("No positive cases exist for " + LABEL + ", returning all unfiltered cases")
        #     else:
        #         print("cannot filter on finding " + finding +
        #               " as not in data - please check spelling")

        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
        self.RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pic = pydicom.read_file(
            os.path.join(
                self.path_to_images,
                self.df.index[idx] + '.dcm'))
        image_2d = pic.pixel_array
        max_val = np.max(image_2d)
        # Rescaling greyscale between 0-255
        image_2d_scaled = []
        for row in image_2d:
            row_scaled = []
            for col in row:
                col_scaled = int((float(col) / float(max_val)) * 255.0)
                row_scaled.append(col_scaled)
            image_2d_scaled.append(row_scaled)
        image_2d_scaled = np.array(image_2d_scaled)
        image = Image.fromarray(image_2d_scaled.astype('uint8'), 'L')
        image = image.convert('RGB')

        label = self.df['Target'].iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    cxr = CXRDataset(
        path_to_images = './train',
        transform=None,
        sample=0,
        finding="any",
        path_to_labels=None)

    cxr[10][0]
