import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

from data import Spectrum, SpectrumRead, SpectrumPreprocess, error_range

Amino2Mass = {'A': 71.03711,
               'B': 0.0,
               'C': 103.0092,
               'D': 115.0296,
               'E': 129.0426,
               'F': 147.0684,
               'G': 57.02146,
               'H': 137.0589,
               'I': 113.0841,
               'J': 0.0,
               'K': 128.095,
               'L': 113.0841,
               'M': 131.0405,
               'N': 114.0429,
               'P': 97.05276,
               'Q': 128.0586,
               'R': 156.101,
               'S': 87.03203,
               'T': 101.0477,
               'U': 0.0,
               'V': 99.06841,
               'W': 186.0793,
               'X': 0.0,
               'Y': 163.0633,
               'Z': 0.0}

# you can expand the modification direction as needed.
Modification2Mass = {'Carbamidomethyl_C': 57.021459,
                     'Oxidation_M': 15.994910,
                     'Carbamyl_C': 43.005809}

# if the mass difference between two peaks larger than MAX_EDGE_MASS, we will skip the computation.
MAX_EDGE_MASS = 250.0

# when we fix the tag length to 5, the corresponding tag feature dimension is 6+5+6 = 17.
class Tag(object):
    def __init__(self, tag_len=5):
        self.sequence_letter = ['A']*tag_len
        self.intensity_list = [1.1]*(tag_len+1)
        self.error_list = [0.1]*tag_len
        self.relevance_degree = [1]*(tag_len+1)
        self.length = 0


# The edge class for spectrum graph.
class Edge(object):
    def __init__(self):
        self.amino_acid = 'A'
        self.modification = 'Carbamidomethyl_C'
        self.modified_flag = True
        self.error = 0.0
        self.from_peak = 0
        self.to_peak = 0


class TagGenerator(object):
    def __init__(self, tag_len=3, error=0.01):
        self.error = error
        self.tag_len = tag_len
        self.tag_list = []

    def reset(self):
        self.tag_list = []
    
    # generate tag set for each specturm conditioned on pre-defined error
    def gen_tag(self, spectrum):
        edges = []
        for i in range(len(spectrum.peaks)):
            edges.append([])
        #edges = [[]]*len(spectrum.peaks)
        relevance = [0]*len(spectrum.peaks)

        # create the edge graph for later graph search
        for i in range(0, len(spectrum.peaks)):
            for j in range(i+1, len(spectrum.peaks)):
                mass_dif = spectrum.peaks[j][0] - spectrum.peaks[i][0]
                if mass_dif > MAX_EDGE_MASS:
                    break
                edge_list = self._mass_difference_to_amino_acid(mass_dif, i, j)
                if edge_list:
                    # print(i, len(edge_list))
                    for e in edge_list:
                        edges[i].append(e)
                        # update relevance degree for the peaks
                        relevance[e.from_peak] += 1
                        relevance[e.to_peak] += 1
        # for e in edges:
        #    print(len(e))
        self.reset()
        # print(len(self.tag_list))
        for i in range(0, len(spectrum.peaks)):
            # print(i)
            tag = Tag(self.tag_len)
            tag.intensity_list[0] = spectrum.peaks[i][0]
            tag.relevance_degree[0] = relevance[i]
            self._dfs_search(spectrum, relevance, edges, i, tag)
        self.tag_list = list(set(self.tag_list))
        return self.tag_list

    # given the mass difference between two peaks, search the candidate amino acid
    def _mass_difference_to_amino_acid(self, mass_difference, from_peak, to_peak):
        edge_list = []
        # small_value, big_value = error_range(mass_difference, self.error)
        for i in Amino2Mass:
            a_diff = abs(mass_difference - Amino2Mass[i])
            if a_diff < self.error :
                e = Edge()
                e.amino_acid = i
                e.modified_flag = False
                e.error = a_diff
                e.from_peak = from_peak
                e.to_peak = to_peak
                edge_list.append(e)
            
            # consider the effect of modification.
            
            for j in Modification2Mass:
                a_diff = abs(mass_difference - Amino2Mass[i] - Modification2Mass[j])
                if mass_difference < self.error:
                    e = Edge()
                    e.amino_acid = i
                    e.modified_flag = True
                    e.modification = j
                    e.error = a_diff
                    e.from_peak = from_peak
                    e.to_peak = to_peak
                    edge_list.append(e)
            
        return edge_list

    # employ the DFS algorithm to search the candidate tag sets for each spectrum
    def _dfs_search(self, spectrum, relevance, edges, current_pos, tag):
        if tag.length == self.tag_len:
            self.tag_list.append(tag)
            tag_sequence = ""
            # in order to store a string of spectrum-tag pair.
            for letter in tag.sequence_letter:
                tag_sequence += letter
            # print(tag_sequence)
            return

        for e in edges[current_pos]:
            tag.sequence_letter[tag.length] = e.amino_acid
            tag.error_list[tag.length] = e.error
            tag.intensity_list[tag.length+1] = spectrum.peaks[e.to_peak][1]
            tag.relevance_degree[tag.length+1] = relevance[e.to_peak]
            tag.length += 1
            self._dfs_search(spectrum, relevance, edges, e.to_peak, tag)
            tag.length -= 1


# Discriminate the tag candidates on the basis of tag features.
class TagDiscriminator(nn.Module):
    def __init__(self, tag_len=5):
        super(TagDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3*tag_len+2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def predict(self, x):
        pred = F.softmax(self.forward(x), dim=1)
        ans = []
        for t in pred:
            if t[0] + 0.5 > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

# The improved loss for tag discriminator.
# This is a combination of XE loss and spectrum peak ratio
class ImprovedLoss(nn.Module):

    def __init__(self,  xi=0.2):
        self.xi = xi

    def forward(self, predict_label, true_label, spectrum_info):
        tag_number = spectrum_info[0]
        peak_number = spectrum_info[1]
        n = tag_number / (peak_number * peak_number)
        loss = true_label * math.log(predict_label) + (1 - true_label) * math.log(1 - predict_label)
        return -loss + self.xi * n

    def update_xi(self, xi):
        self.xi = xi

if __name__ == '__main__':
    mgf_file_path="./data/mouse.mgf"
    sp = SpectrumRead(mgf_file_path)
    spectra_list = sp.read_spectrum_data()
    i = 0
    for j in range(len(spectra_list)):
        process = SpectrumPreprocess(200)
        s = process.spectrum_preprocess(spectra_list[i])
        # print(s.peaks)
        tag_generator = TagGenerator(tag_len=3)
        tag_list = tag_generator.gen_tag(s)
        print('len:', len(tag_list))
        print(i)
        i += 1
#    for tag in tag_list:
#        print(tag.sequence_letter)
    print('done')
