import torch
import argparse
from model import TagDiscriminator, TagGenerator
from data import Spectrum, SpectrumRead, SpectrumPreprocess

# ------------------------------------------------------------------
# parameter setting
# path to the CKPT
model_path = './data/CKPT'
# path to the mgf which we want to extract tag
mgf_path = './data/human.mgf'
# path to store the results
res_path = './data/res.txt'
# ------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_init(model_path, device):
    ckpt = torch.load(model_path, map_location='cpu')
    discriminator = TagDiscriminator()
    discriminator.load_state_dict(ckpt['model'])
    discriminator = discriminator.to(device)
    discriminator.eval()
    return discriminator


if __name__ == '__main__':
    
    # intialize the parameter
    parser = argparse.ArgumentParser(description='GameTag')
    parser.add_argument('--mgf_path', type=str, default=mgf_path)
    parser.add_argument('--model_path', type=str, default=model_path)
    parser.add_argument('--res_path', type=str, default=res_path)
    args = parser.parse_args()

    print('Thank you for using GameTag Tool!')
    # read the mgf file and get the spectra list according to given path
    print('read mgf file ...')
    spectrum_read = SpectrumRead(args.mgf_path)
    spectra_list = spectrum_read.read_spectrum_data()
    print('Total spectrum number: ', len(spectra_list))

    # deal with each spectrum
    total_num = 0
    correct_num = 0
    for spectrum in spectra_list:
        sp_name = []
        sp_name.append(spectrum.spectrum_id)
        # preprocess each spectrum: 
        # 1. select top intensity peaks
        # 2. remove the isotope peaks
        # 3. transform peak to single charge
        # 4. insert 0, parent peak to list
        process = SpectrumPreprocess(top_peaks_num=200)
        refined_spectrum = process.spectrum_preprocess(spectrum)

        # generate the tag candidates for each spectrum
        tag_generator = TagGenerator(tag_len=5)
        tag_list = tag_generator.gen_tag(refined_spectrum)

        # load the tag discriminator
        tag_discriminator = model_init(args.model_path, device)

        # discriminate the tag candidates according to corresponding features
        total_num += 1
        for tag in tag_list:
            tag_feature = tag.intensity_list + tag.error_list + tag.relevance_degree
            tag_feature = torch.FloatTensor([tag_feature]).to(device)

            correct = tag_discriminator.predict(tag_feature).item()
            if correct == 0:
                tag_sequence = ""
                for letter in tag.sequence_letter:
                    tag_sequence += letter
                sp_name.append(tag_sequence)
                # print(tag_sequence)
        print('\r'+ 'current number / total number: {:d} / {:d}'.format(total_num, len(spectra_list)),\
            end='', flush=True)
        #print(sp_name)
        name_str = ""
        for sp in sp_name:
            name_str = name_str + sp + '\n'
        with open(res_path, 'a') as f:
            f.write(name_str)
    print(' ')
    print('The results can be found in:', res_path)


