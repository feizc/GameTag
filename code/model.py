import torch.nn as nn
import os
import pickle
import math
import sys
import time
from data_preprocess import Spectrum, SpectrumRead

mgf_file_path = "../data/20100826_Velos2_AnMi_SA_HeLa_4Da_HCDFT.mgf"  # the path of mgf file
tag_feature_save_dir = "../data"
spectrum_results_file_path = '../data/human.txt'

Amino2Mass = {'A': 71.03711,
               'B': 111.032,
               'C': 103.0092,
               'D': 115.0296,
               'E': 129.0426,
               'F': 147.0684,
               'G': 57.02146,
               'H': 137.0589,
               'I': 113.0841,
               'J': 173.0324,
               'K': 128.095,
               'L': 113.0841,
               'M': 131.0405,
               'N': 114.0429,
               'P': 97.05276,
               'Q': 128.0586,
               'R': 156.101,
               'S': 87.03203,
               'T': 101.0477,
               'U': 113.0841,
               'V': 99.06841,
               'W': 186.0793,
               'X': 113.0477,
               'Y': 163.0633,
               'Z': 103.0029}

# you can expand the modification direction as needed.
Modification2Mass = {'Carbamidomethyl_C': 57.021459,
                     'Oxidation_M': 15.994910,
                     'Acetyl_K': 42.010568,
                     'Carbamyl_C': 43.005809}

# if the mass difference between two peaks larger than MAX_EDGE_MASS, we will skip the computation.
MAX_EDGE_MASS = 300.0


# when we fix the tag length to 5, the corresponding tag feature dimension is 6+5+6 = 17.
class Tag(object):
    def __init__(self):
        self.sequence_letter = ['A']*5
        self.intensity_list = [1.1]*6
        self.error_list = [0.1]*5
        self.relevance_degree = [1]*6
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


# Discriminate the tag candidates on the basis of tag features.
class Discriminator(nn.Module):
    def __init__(self, tag_length=5):
        super(Discriminator, self).__init__()
        self.input_dim = 3 * tag_length + 2
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# generate the tag candidates conditioned on each mass spectrum.
class Generator(object):
    def __init__(self, spectra_list, error=0.1):
        self.error = error
        self.tag_list = []  # extracted Tag class for each spectrum.
        self.spectrum_tag = []  # pair of spectrum-tag.
        self.spectra_list = spectra_list
        self.tag_length = 5

    # Given the mass difference between two peaks, search the candidate amino acid.
    def mass_difference_to_amino_acid(self, mass_difference, from_peak, to_peak):
        edge_list = []
        for i in Amino2Mass:
            mass_error = abs(mass_difference - Amino2Mass[i])
            if mass_error < self.error:
                e = Edge()
                e.amino_acid = i
                e.modified_flag = False
                e.error = mass_error
                e.from_peak = from_peak
                e.to_peak = to_peak
                edge_list.append(e)

            # consider the effect of modification.
            for j in Modification2Mass:
                mass_error = abs(mass_difference - Amino2Mass[i] - Modification2Mass[j])
                if mass_difference < self.error:
                    e = Edge()
                    e.amino_acid = i
                    e.modification = j
                    e.error = mass_error
                    e.from_peak = from_peak
                    e.to_peak = to_peak
                    edge_list.append(e)
        return edge_list

    # Employ the DFS algorithm to search the candidate tag sets for each spectrum.
    def _dfs_search(self, spectrum, relevance, edges, current_pos, tag, tag_set):
        if tag.length == self.tag_length:
            tag_set.append(tag)
            tag_sequence = ""
            # in order to store a string of spectrum-tag pair.
            for letter in tag.sequence_letter:
                tag_sequence += letter
            self.spectrum_tag.append(spectrum.spectrum_id + ' ' + tag_sequence)
            print('\r' + tag_sequence + '           ', end='', flush=True)
            # time.sleep(0.3)
            return
        for e in edges[current_pos]:
            tag.sequence_letter[tag.length] = e.amino_acid
            tag.error_list[tag.length] = e.error
            tag.intensity_list[tag.length+1] = spectrum.peaks[e.to_peak][1]
            tag.relevance_degree[tag.length+1] = relevance[e.to_peak]
            tag.length += 1
            self._dfs_search(spectrum, relevance, edges, e.to_peak, tag, tag_set)
            tag.length -= 1

    # generate tag set for each specturm conditioned on pre-defined error.
    def gen_tag(self, spectrum):
        edges = [[]]*len(spectrum.peaks)
        relevance = [0]*len(spectrum.peaks)
        # Create the edge graph for later graph search.
        for i in range(0, len(spectrum.peaks)):
            for j in range(i+1, len(spectrum.peaks)):
                mass_difference = spectrum.peaks[j][0]-spectrum.peaks[i][0]
                if mass_difference > MAX_EDGE_MASS:
                    break
                edge_list = self.mass_difference_to_amino_acid(mass_difference, i, j)
                if edge_list:
                    for e in edge_list:
                        edges[i].append(e)
                        # update the relevance degree for the peaks.
                        relevance[e.from_peak] += 1
                        relevance[e.to_peak] += 1
        # print(relevance)
        tag_set = []
        # begin from each peak to enumerate the tag sequence candidates.
        for i in range(0, len(spectrum.peaks)):
            tag = Tag()
            tag.intensity_list[0] = spectrum.peaks[i][1]
            tag.relevance_degree[0] = relevance[i]
            self._dfs_search(spectrum, relevance, edges, i, tag, tag_set)
        return tag_set

    # process the entire spectrum list.
    def process_spectrum_list(self):
        for spectrum in self.spectra_list:
            # print(spectrum.spectrum_id)
            self.tag_list.append(self.gen_tag(spectrum))

    # Save the extracted tag sequence and features.
    def save_tag_feature(self, path):
        tag_feature = []
        # directly concatenate the features of each tag as one vector.
        for tag_set in self.tag_list:
            for tag in tag_set:
                per_tag_feature = tag.intensity_list + tag.error_list + tag.relevance_degree
                tag_feature.append(per_tag_feature)
        with open(os.path.join(tag_feature_save_dir, 'tag_feature_dataset.pkl'), 'wb') as f:
            pickle.dump(tag_feature, f)
        # we also save the spectrum-tag pair for performance comparison conveniently.
        with open(os.path.join(tag_feature_save_dir, 'spectrum_tag.txt'), 'w') as f:
            for line in self.spectrum_tag:
                f.write(line + '\n')


# select the mass spectrum only find in the identification results.
# Only employed during training.
def spectrum_refine(spectra_list, spectrum_results_file_path):
    spectrum_peptide_match = dict()
    refined_spectra_list = []

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

    # store the interaction.
    for spectrum in spectra_list:
        if spectrum.spectrum_id in spectrum_peptide_match.keys():
            refined_spectra_list.append(spectrum)

    return refined_spectra_list


# transfer a peak of M with parent mass P to a node M-|H| (b fragments) and P-M (y fragments)
def peak_to_node(spectrum):
    node = []
    parent_mass = spectrum.pep_mass

    # tow addition auxiliary node for graph.
    node.append([0, 0])
    node.append([spectrum.pep_mass, 0])

    # spectrum.peaks -> [mass, intensity]
    for peak in spectrum.peaks:
        node.append([peak[0]-1, peak[1]])
        node.append([parent_mass-peak[0],peak[1]])
    return node.sort()


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


'''
# Module Test: tag generation process.

s = SpectrumRead(mgf_file_path)
spectra_list = s.read_spectrum_data()
print("Read the mgf file successfully!")
print("The program is extracting the tag sequence conditioned on each spectrum as: ")
t = Generator(spectra_list, 0.1)
t.process_spectrum_list()
print("The extracted tag sequences are saved!")
t.save_tag_feature(tag_feature_save_dir)
'''
