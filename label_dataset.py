import os
import pickle

from data import Spectrum, SpectrumRead, SpectrumPreprocess, error_range
from model import TagGenerator

#-----------------------------------------------------------------------
# parameter setting
# the path to mgf file
mgf_file_path = "./data/mouse.mgf"
# the results file of the open-pfind, peaks and msfragger
psm_path = "./data/mouse.txt"
# the path to store the labeled data
labeled_path = "./data/labeled_tag.pkl"
#----------------------------------------------------------------------

# read the labeled results from open-pfind, MSfragger and Peaks
psm_dic = {}
with open(psm_path, 'r') as f:
    lines = f.readlines()
    print(len(lines)-1)
    for i in range(1, len(lines)):
        line = lines[i].split('\t')
        spectrum_id = line[0]
        pep_sequence = line[5]
        psm_dic[spectrum_id] = pep_sequence

#print(psm_dic.keys())
#print(psm_dic.values())

# read the mgf file
spectrum_read = SpectrumRead(mgf_file_path)
spectra_list = spectrum_read.read_spectrum_data()

# generate the tag candidates
labeled_data = []
sp_name = []
correct = 0
total = 0
correct_num = 1
for i in range(len(spectra_list)):
    if psm_dic.__contains__(spectra_list[i].spectrum_id):
        pep_sequence = psm_dic[spectra_list[i].spectrum_id]
        pep_sequence = pep_sequence.replace('I','L')
        # print(pep_sequence)
    else:
        continue
    flag = False
    process = SpectrumPreprocess(200)
    refined_spectrum = process.spectrum_preprocess(spectra_list[i])
    # print(refined_spectrum.peaks)
    tag_generator = TagGenerator(tag_len=5)
    tag_list = tag_generator.gen_tag(refined_spectrum)


    for tag in tag_list:
        tag_feature = tag.intensity_list + tag.error_list + tag.relevance_degree
        tag_sequence = ""
        for letter in tag.sequence_letter:
            tag_sequence += letter
        tag_sequence = tag_sequence.replace('I','L')
        tag_sequence_r = tag_sequence[::-1]
        if tag_sequence in pep_sequence or tag_sequence_r in pep_sequence:
            tag_feature += [1, 0]
            correct_num += 1
            labeled_data.append(tag_feature)
            flag = True
            # print('correct! ', tag_sequence)
        else:
            tag_feature += [0, 1]
            if correct_num > 0:
                labeled_data.append(tag_feature)
                correct_num -= 1
    if flag == True:
        correct += 1
        sp_name.append(spectra_list[i].spectrum_id)
        #print(tag_sequence)
        #print(len(tag_feature))
    print('\r'+'correct / current / total: {:d} / {:d} / {:d}'.format(correct, i+1, len(spectra_list)), end='', flush=True)

name_str = ""
for sp in sp_name:
    name_str = name_str + sp + '\n'
with open("./data/sp_name.txt", 'w') as f:
    f.write(name_str)

#print(len(labeled_data))
#print(labeled_data[0])
with open(labeled_path, 'wb') as f:
    pickle.dump(labeled_data, f)

