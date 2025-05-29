import os
from torch.utils.data import Dataset
import torch
import csv
MAX_IC = 1000.0
WTH = 'QITLKESGPTLVKPTQTLTLTCTFSGFSLSISGVGVGWLRQPPGKALEWLALIYWDDDKRYSPSLKSRLTISKDTSKNQVVLKMTNIDPVDTATYYCAHHSISTIFDHWGQGTLVTVSS'
WTL = 'QSALTQPASVSGSPGQSITISCTATSSDVGDYNYVSWYQQHPGKAPKLMIFEVSDRPSGISNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTTSSAVFGGGTKLTVL'
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

def mutate(seq, mutations):
    for mutation in mutations:
        wt ,site, mutation = str(mutation[0]), int(mutation[1:-1]), str(mutation[-1])
        site -= 1
        assert seq[site] == wt, f"site {site} mistmatch with WT {wt} {seq[site-2:site+2]}"
        seq = seq[:site] + mutation + seq[site+1:]  # 在指定 site 上进行氨基酸突变
    return seq

def read_csv_to_list(file_path,class_name = 'BQ.1.1'):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row
        for i, row in enumerate(reader):
            # print(row)
            mutation_chain = row[2]
            mutation = row[3]
            if mutation_chain == 'HC':
                # mutation: T70S -> T,70,S or T70S_P10V -> [(T,70,S),(P,10,V)]
                if '_'  in mutation:
                    mutations = mutation.split('_')
                else:
                    mutations = [mutation]
                vh_seq = mutate(WTH,mutations)
                vl_seq = WTL
            elif mutation_chain == 'LC':
                if '_'  in mutation:
                    mutations = mutation.split('_')
                else:
                    mutations = [mutation] 
                vh_seq = WTH
                vl_seq = mutate(WTL, mutations)
            elif mutation_chain == 'both':
                H_mutations, L_mutations = mutation.split('/')
                mutations = []
                if '_'  in L_mutations:
                    L_mutations = L_mutations.split('_')
                else:
                    L_mutations = [L_mutations]
                if '_'  in H_mutations:
                    H_mutations = H_mutations.split('_')
                else:
                    H_mutations = [H_mutations]
                vh_seq = mutate(WTH,H_mutations)
                vl_seq = mutate(WTL,L_mutations)
            else:
                vh_seq = WTH
                vl_seq = WTL
            if class_name == 'BA.1' or class_name == 'WT':
                text = row[4]
            elif class_name == 'BQ.1.1':
                text = row[-3]
            if '>' in text:
                ic50 = MAX_IC
            else:
                number_part = text.split("±")[0].strip()  # 分割并移除额外的空格
                ic50 = float(number_part)  # 转换为浮点数
            print(ic50)
            result.append({
                'H': vh_seq,
                'L': vl_seq,
                'virus': class_name,
                'IC50': ic50,
                # 'H_CDR3':row[-4],
                # 'L_CDR3':row[-1]
            })
    return result

class IC50DatasetLY(Dataset):
    def __init__(self, spike_data = './data/Ag_sequence.csv', ic50_data = './data/Ly1404_SILM.csv'):
        super().__init__()
        self.ic_50_data = read_csv_to_list(ic50_data)
        self.spike_dict = read_csv_to_dict(spike_data)

    def __len__(self):
        return len(self.ic_50_data)

    def __getitem__(self, idx, use_rbd = False):# {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        res['spike'],res['weight'] = self.spike_dict[res['virus']]
        if use_rbd:
            RBD_start = res['spike'].find('NITN')
            RBD_end = res['spike'].find('KKST') + 4
            res['spike'] = res['spike'][RBD_start:RBD_end + 1]
        return res
    
# dataset = IC50DatasetLY()
# data = dataset[0]
# print(data)