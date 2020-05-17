import os
import torch
import pickle
tag_feature_save_dir = "../data"
# the intersection labeled dataset can be obtained by data_label_union.py
# according to the identification results of PEAKS, MSFragger+ and Open-pFind.
spectrum_results_file_path = '../data/human.txt'


# discriminate the correctness of generated tag sequence according to the results from label_data.
def tag_correct_discriminate(tag_feature_save_dir, spectrum_results_file_path):
    # store each PSMs as directionary.
    spectrum_peptide_match = dict()
    tag_correctness = []

    # read the labeled PSM.
    with open(spectrum_results_file_path, 'r') as f:
        f.readline()
        while True:
            line = f.readline().split()
            if line:
                spectrum_id = line[0]
                peptide_sequence = line[5]
                spectrum_peptide_match[spectrum_id] = peptide_sequence
                # print(spectrum_id, peptide_sequence)
            else:
                break

    # read the generated tag candadates.
    with open(os.path.join(tag_feature_save_dir, 'spectrum_tag.txt'), 'r') as f:
        while True:
            line = f.readline().split()
            if line:
                spectrum_id = line[0]
                tag_sequence = line[1]
                if tag_sequence in spectrum_peptide_match[spectrum_id]:
                    tag_correctness.append([1, 0])
                # reverse the tag sequence to validate the correctness.
                elif tag_sequence[::-1] in spectrum_peptide_match[spectrum_id]:
                    tag_correctness.append([1, 0])
                else:
                    tag_correctness.append([0, 1])
            else:
                break
    # print(tag_correctness)

    # save the correctness of each tag as label for later Discriminator training.
    with open(os.path.join(tag_feature_save_dir, 'tag_label_dataset.pkl'), 'wb') as f:
        pickle.dump(tag_correctness, f)

    # validate no loss of accuracy for feature store.
    # with open(os.path.join(tag_feature_save_dir, 'tag_label_dataset.pkl'), 'rb') as f:
    #    result = pickle.load(f)
    #    print(result)


# keep track of most recent, average, sum and count of a metric.
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# tag_correct_discriminate(tag_feature_save_dir, spectrum_results_file_path)

