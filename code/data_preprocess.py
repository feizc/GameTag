# mgf_file_path="../data/20100826_Velos2_AnMi_SA_HeLa_4Da_HCDFT.mgf" # the path of mgf file


# Class for mass spectrum, we store attribute id, parent_charge, pepmass, mass, intensity for each spectrum.
class Spectrum(object):
    def __init__(self):
        self.peaks = []  # peaks pair [mass, intensity]
        self.spectrum_id = "HeLa"
        self.parent_charge = 1
        self.pep_mass = 1.0

    def update_parent_charge(self,charge):
        self.parent_charge = charge

    def update_spectrum_id(self, spectrum_id):
        self.spectrum_id = spectrum_id[:-1]

    def update_pep_mass(self, pep_mass):
        self.pep_mass = pep_mass

    def add_peak(self, x, y):
        self.peaks.append([x,y])

    def reset(self):
        self.peaks = []

    # Transform mass-to-charge into mass
    def eliminate_charge_influence(self):
        self.pep_mass = self.pep_mass * self.parent_charge
        for i in range(len(self.peaks)):
            self.peaks[i][0] = self.peaks[i][0] * self.parent_charge

    # Data preprocess: select top-n intensity peaks for each mass spectrum.
    def data_preprocess(self, max_peaks_num):
        if len(self.peaks) > max_peaks_num:
            self.peaks.sort(key=lambda x: x[1], reverse=True)
            self.peaks = self.peaks[:max_peaks_num]
            self.peaks.sort()


# Class for spectrum reading.
class SpectrumRead(object):
    def __init__(self, file_path, max_peaks_num=500):
        self.file_path = file_path
        self.max_peaks_num = max_peaks_num
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
                tmp_spectrum.eliminate_charge_influence()
                tmp_spectrum.data_preprocess(self.max_peaks_num)
                # Print modified mass spectrum data
                # print(tmp_spectrum.peaks)
                # print(len(tmp_spectrum.peaks))
                self.spectrum_data.append(tmp_spectrum)
            elif line[0] == 'P':
                idx = line.find('=')
                pep_mass = line[idx+1:]
                tmp_spectrum.update_pep_mass(float(pep_mass))
            else:
                tmp_list = line.split()
                tmp_spectrum.add_peak(float(tmp_list[0]), float(tmp_list[1]))
        return self.spectrum_data




# s = SpectrumRead(mgf_file_path)
# spectra_list = s.read_spectrum_data()
# print(spectra_list[0].spectrum_id)
