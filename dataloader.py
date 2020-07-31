import torch
from torch.utils.data import Dataset
import pickle


# A Pytorch Dataset class to be used in a Pytorch DataLoader to create batches
class TagFeatureLoader(Dataset):
    def __init__(self, tag_feature_path):
        # load labebed data [num_tag, 19 = [17 , 2]]
        with open(tag_feature_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, i):
        feature = torch.FloatTensor(self.data[i][:17])
        label = torch.LongTensor(self.data[i][17:])
        return feature, label

    def __len__(self):
        return len(self.data)