import os
from torch.utils.data import Dataset
import torch
import csv
MAX_IC = 10.0

def parse_weight_vector(weight_vector_str):
    return [1.0] + [float(weight) for weight in weight_vector_str[1:-1].split(',')] + [1.0]

def read_csv_to_dict(file_path):
    result = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            col_1, col_2,col_3 = row[0], row[2],parse_weight_vector(row[3])
            result[col_1] = (col_2,torch.tensor(col_3))
    return result

def read_csv_to_list(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row

        for i, row in enumerate(reader):
            vh_seq = row[-2]
            vl_seq = row[-1]
            epitope = row[1]
            ic50 = 10.0*(i+1)
            if epitope == 'E3':
                result.append({
                    'H': vh_seq,
                    'L': vl_seq,
                    'virus': 'RBD',
                    'IC50': ic50,
                    # 'H_CDR3':row[-4],
                    # 'L_CDR3':row[-1]
                })
    return result

class IC50DatasetDMS(Dataset):
    def __init__(self, spike_data = './data/Ag_sequence.csv', ic50_data = './data/antibody_info.csv'):
        super().__init__()
        self.ic_50_data = read_csv_to_list(ic50_data)
        self.spike_dict = read_csv_to_dict(spike_data)

    def __len__(self):
        return len(self.ic_50_data)

    def __getitem__(self, idx, use_rbd = False):# {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        res['spike'],res['weight'] = self.spike_dict['RBD']
        if use_rbd:
            RBD_start = res['spike'].find('NITN')
            RBD_end = res['spike'].find('KKST') + 4
            res['spike'] = res['spike'][RBD_start:RBD_end + 1]
        return res
