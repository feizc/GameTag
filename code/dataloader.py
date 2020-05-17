import torch
from torch.utils.data import Dataset
import pickle


# A Pytorch Dataset class to be used in a Pytorch DataLoader to create batches.
class SpectrumLoader(Dataset):
    def __init__(self, tag_feature_file_path, tag_label_file_path):
        # load tag features [num_tag, 17].
        with open(tag_feature_file_path, 'rb') as f:
            self.tag_feature = pickle.load(f)

        # load tag label [num_tag, 2]. two classes for two-classification: right and wrong
        with open(tag_label_file_path, 'rb') as f:
            self.tag_label = pickle.load(f)

        assert len(self.tag_feature[0]) == len(self.tag_label)

    def __getitem__(self, i):
        tag_feature = torch.FloatTensor(self.tag_feature[i])
        tag_label = torch.Tensor(self.tag_label[i])
        return tag_feature, tag_label

    def __len__(self):
        return len(self.tag_label)

