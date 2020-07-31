# Class for mass spectrum, we store attribute id, parent_charge, pepmass, mass, intensity for each spectrum.
import copy
class Spectrum(object):
    def __init__(self):
        self.peaks = []  # peaks pair [mass, intensity, charge=1]
        self.spectrum_id = "HeLa"
        self.parent_charge = 1
        self.pep_mass = 1.0

    def update_parent_charge(self,charge):
        self.parent_charge = charge

    def update_spectrum_id(self, spectrum_id):
        self.spectrum_id = spectrum_id[:-1]

    def update_pep_mass(self, pep_mass):
        self.pep_mass = pep_mass

    def add_peak(self, mass, intensity, charge):
        self.peaks.append([mass, intensity, charge])

    def reset(self):
        self.peaks = []


# Class for spectrum reading from mgf file.
class SpectrumRead(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.spectrum_data = []

    # Read .mgf file for each mass spectrum.
    def read_spectrum_data(self):
        tmp_spectrum = Spectrum()
        for line in open(self.file_path):
            if line[0] == 'B':
                tmp_spectrum.reset()
            elif line[0] == 'T':
                idx = line.find('=')
                spectrum_id = line[idx+1:]
                tmp_spectrum.update_spectrum_id(spectrum_id)
            elif line[0] == 'C':
                idx = line.find('=')
                parent_charge = line[idx+1]
                tmp_spectrum.update_parent_charge(int(parent_charge))
            elif line[0] == 'R':
                continue
            elif line[0] == 'E':
                # end of current spectrum
                self.spectrum_data.append(copy.deepcopy(tmp_spectrum))
            elif line[0] == 'P':
                idx = line.find('=')
                pep_mass = line[idx+1:]
                tmp_spectrum.update_pep_mass(float(pep_mass))
            else:
                tmp_list = line.split()
                # we set the charge of peak as 1 by default
                tmp_spectrum.add_peak(float(tmp_list[0]), float(tmp_list[1]), 1)
        return self.spectrum_data

# calculate the ppm error range
def error_range(value, ppm=20.0):
    small_value = value * 1000000.0 / (1000000.0 + ppm)
    big_value = value * 1000000.0 / (1000000.0 - ppm)
    return small_value, big_value

# preprocess each spectrum from list:
# 1. select top intensity peaks
# 2. remove the isotope peaks
# 3. transform peak to single charge
# 4. insert 0, parent peak to list
class SpectrumPreprocess(object):
    def __init__(self, top_peaks_num=200):
        super(SpectrumPreprocess, self).__init__()
        self.top_peaks_num = top_peaks_num

    # select top-n intensity peaks for each mass spectrum.
    def select_top_n(self, spectrum):
        if len(spectrum.peaks) > self.top_peaks_num:
            spectrum.peaks.sort(key=lambda x: x[1], reverse=True)
            spectrum.peaks = spectrum.peaks[:self.top_peaks_num]
            spectrum.peaks.sort()
        max_intensity = -1.0
        for peak in spectrum.peaks:
            if peak[1] > max_intensity:
                max_intensity = peak[1]
        for i in range(len(spectrum.peaks)):
            spectrum.peaks[i][1] = spectrum.peaks[i][1] / max_intensity
        return spectrum

    def remove_isotope(self, spectrum):
        flag = [0] * len(spectrum.peaks)
        current_peaks = spectrum.peaks
        # loop for determining the charge and isotope
        while True:
            idx = -1
            current_max_intensity = -1.0
            for i in range(len(current_peaks)):
                # this peak has been determined
                if flag[i] == 1:
                    continue
                else:
                    if current_peaks[i][1] > current_max_intensity:
                        idx = i 
                        current_max_intensity = current_peaks[i][1]
            # all peaks have been proposed
            if idx == -1:
                break

            flag[idx] = 1
            # determine the current peak charge number
            if spectrum.pep_mass == 1.0:
                charge_max = 1
            else:
                charge_max = int(spectrum.parent_charge)-1
            # find the isotope assume the charge to 1, 2,..., parent_charge-1
            for charge in range(1, charge_max+1):
                find_flag = False
                if idx < len(current_peaks)-1 and flag[idx+1] == 0:
                    right_dis = current_peaks[idx+1][0] - current_peaks[idx][0]
                    # if the next peak is isotope, we calculate the peak mass
                    predict_value = current_peaks[idx][0] + 1.0 / charge
                    small_value, big_value = error_range(predict_value)
                    if current_peaks[idx+1][0] >= small_value and current_peaks[idx+1][0] <= big_value:
                        flag[idx+1] = 1
                        current_peaks[idx+1][1] = -1.0
                        current_peaks[idx][2] = charge
                        find_flag = True
                if idx > 1 and flag[idx-1] == 0:
                    left_dis = current_peaks[idx][0] - current_peaks[idx-1][0]
                    predict_value = current_peaks[idx][0] - 1.0 / charge
                    small_value, big_value = error_range(predict_value)
                    if current_peaks[idx-1][0] >= small_value and current_peaks[idx-1][0] <= big_value:
                        flag[idx-1] = 1
                        current_peaks[idx-1][1] = -1.0
                        current_peaks[idx][2] = charge
                        find_flag = True
                if find_flag == True:
                    break
        # remove the isotype peaks
        refined_peaks = []
        for peak in current_peaks:
            if peak[1] == -1.0:
                continue
            else:
                refined_peaks.append(peak)
        spectrum.peaks=refined_peaks
        return spectrum

    # transform all peak to single charge and insert 0, parent peak
    def transform_to_single_charge(self, spectrum):
        for peak in spectrum.peaks:
            if peak[2] != 1:
                peak[0] = peak[0] * peak[2] - (peak[2] - 1) * 1.0
        spectrum.peaks.insert(0, [0, 0, 1])
        p = spectrum.parent_charge * spectrum.pep_mass - (spectrum.parent_charge - 1) * 1.0
        spectrum.peaks.append([p, 0, spectrum.parent_charge])
        return spectrum

    def spectrum_preprocess(self, spectrum):
        spectrum = self.select_top_n(spectrum)
        spectrum = self.remove_isotope(spectrum)
        spectrum = self.transform_to_single_charge(spectrum)
        spectrum.peaks.sort()
        return spectrum


if __name__ == '__main__':
    mgf_file_path="./data/mouse.mgf" # the path of mgf file
    s = SpectrumRead(mgf_file_path)
    spectra_list = s.read_spectrum_data()
    process = SpectrumPreprocess(200)
    print(len(spectra_list))
    for sp in spectra_list:
        sp = process.spectrum_preprocess(sp)
        print(sp.peaks)
    '''
    s = process.spectrum_preprocess(spectra_list[0])
    print(s.peaks)
    print(len(s.peaks))
    s = process.spectrum_preprocess(spectra_list[1])
    print(s.peaks)
    print(len(s.peaks))
    s = process.spectrum_preprocess(spectra_list[2])
    print(s.peaks)
    print(len(s.peaks))
    '''
