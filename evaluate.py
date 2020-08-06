import os

mgf_path = '20100826_Velos2_AnMi_SA_HeLa_4Da_HCDFT.mgf'
psm_path = 'human.txt'
res_path = '20100826_Velos2_AnMi_SA_HeLa_4Da_HCDFT.tags'

def mgf_read(mgf_path):
    sp_list = []
    for line in open(mgf_path):
        if line[0] == 'T':
            idx = line.find('=')
            sp_id = line[idx+1:-1]
            sp_list.append(sp_id)
    return sp_list

def psm_read(psm_path):
    psm_dic = {}
    with open(psm_path, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i].split('\t')
            sp_id = line[0]
            sp_s = line[5]
            psm_dic[sp_id] = sp_s
    return psm_dic

def idx_compute(sp_list, sp_dic):
    idx_list = []
    for i in range(len(sp_list)):
        if sp_dic.__contains__(sp_list[i]):
            idx_list.append(i)
    return idx_list

def sensitivity(res_path, idx_list, sp_dic, sp_list):
    correct, total = 0, 0
    tag_num = 0
    pep_seq = ""
    flag =False
    c_flag = False
    for line in open(res_path, 'r'):
        if line[0] == 'H':
            continue
        elif line[0] == 'S':
            if c_flag == True:
                correct += 1
                c_flag = False
            line = line.split('\t')[1]
            idx = line.find('=')
            line = int(line[idx+1:])
            if line in idx_list:
                flag = True
                pep_seq = sp_dic[sp_list[line]].replace('I','L')
                total += 1
            else:
                flag = False
        elif line[0] == 'T':
            if flag == False:
                continue
            else:
                tag_num += 1
                line = line.split('\t')[1].replace('I','L')
                if line in pep_seq:
                    c_flag = True
                line = line[::-1]
                if line in pep_seq:
                    c_flag = True
    return correct, total, tag_num


if __name__ == '__main__':

    sp_list = mgf_read(mgf_path)
    sp_dic = psm_read(psm_path)
    # print(sp_dic.keys())
    idx_list = idx_compute(sp_list, sp_dic)
    # print(idx_list)
    print(len(sp_list))

    correct, total, tag_num = sensitivity(res_path, idx_list, sp_dic, sp_list)
    print(correct)
    print(total)
    print(tag_num)

