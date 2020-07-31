def name_list_read(path):
    f = open(path, 'r')
    name_list = f.read().split('\n')[:-1]
    f.close()
    return name_list

def mgf_intersection(mgf_path, res_path, name_list):
    spectrum_inf = []
    i = 0
    for line in open(mgf_path):
        if line[0] == 'E':
            spectrum_inf.append(line)
            spectrum_id = spectrum_inf[1]
            idx = spectrum_id.find('=')
            spectrum_id = spectrum_id[idx+1:-1]
            print(spectrum_id)
            if spectrum_id in name_list:
                with open(res_path, 'a+') as f:
                    for l in spectrum_inf:
                        f.write(l)
            spectrum_inf.clear()
            i += 1
            print(i, 'th spectrum')
        else:
            spectrum_inf.append(line)


if __name__ == '__main__':
    name_list = name_list_read('sp_name.txt')
    #print(name_list)
    mgf_path = 'mouse.mgf'
    res_path = 'res.mgf'
    mgf_intersection(mgf_path, res_path, name_list)
    #f = open('sp_name.txt', 'r')
    #name_list = f.read().split('\n')
    #print(name_list[:-1])

