import os
from torch.utils.data import Dataset
import torch
import csv
import pandas as pd
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
    WT_seq = 'MRVKGIQMNSQHLLRWGIMILGMIMICSVAGNLWVTVYYGVPVWKDAETTLFCASDAKAYDAEVHNIWATHACVPTDPNPQEINLENVTEEFNMWKNNMVEQMHTDIISLWDQGLKPCVKLTPLCVTLDCHNVTYNITSDMKEEITNCSYNVTTVIRDKKQKVSSLFYKLDVVQIGGNNRTNSQYRLINCNTSAITQACPKVTFEPIPIHYCAPAGFAILKCKDEKFNGTGLCKNVSTVQCTHGIKPVVSTQLLLNGSLAEGEVRIRSENITNNAKNIIVQLASPVTINCIRPNNNTRKSVHLGPGQAFYATDGIIGEIRQAHCNVSKKEWNSTLQKVANQLRPYFKNNTIIKFANSSGGDLEITTHSFNCGGEFFYCNTSGLFNSTWEFNSTWNNSNSTENITLQCRIKQIINMWQRAGQAIYAPPIPGVIRCKSNITGLILTRDGGSNKNTSETFRPGGGDMRDNWRSELYKYKVVKIEPIGVAPTRAKRRVVEREKRAVGIGAVFIGFLGAAGSTMGAASVTLTVQARQLLSGIVQQQSNLLRAIEAQQHLLKLTVWGIKQLQARVLAVERYLKDQQLLGIWGCSGKLICTTNVPWNSSWSNKSQDEIWGNMTWLQWDKEVSNYTQIIYTLIEESQNQQEKNEQDLLALDKWASLWNWFNISQWLWYIKIFIIIVGGLIGLRIVFAVLSVINRVRQGYSPLSFQTRTPNPGELDRPGRIEEEGGEQDRGRSIRLVSGFLALAWDDLRSLCLFSYHRLRDFILIATRTVELLGHSSLKGLRLGWESLKYLGNLLVYWGRELKISAINLCDTIAIAVAGWTDRVIELGQRLCRAILHIPRRIRQGFERALL*',
    
    # escape_data = pd.read_csv('./data/3BNC117_avg.csv')
    site_numbering_data = pd.read_csv('./data/site_numbering_map.csv')
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                escape_v = float(row[5])
            except ValueError:
                continue
            mutated_seq =  mutate_sequence(*WT_seq, row[1], row[2], row[3], site_numbering_data)
            if mutated_seq == None:
                continue
            result.append({
                    'H': 'QVQLLQSGAAVTKPGASVRVSCEASGYNIRDYFIHWWRQAPGQGLQWVGWINPKTGQPNNPRQFQGRVSLTRHASWDFDTYSFYMDLKALRSDDTAVYFCARQRSDYWDFDVWGSGTQVTVSS',
                    'L': 'DIQMTQSPSSLSASVGDTVTITCQANGYLNWYQQRRGKAPKLLIYDGSKLERGVPSRFSGRRWGQEYNLTINNLQPEDIATYFCQVYEFVVPGTRLDLK',
                    'spike': mutated_seq,
                    'IC50': escape_v,# mean escape
                })
    return result


# Define the mutate_sequence function again for execution context
def mutate_sequence(sequence, site, wildtype, mutant, site_numbering_data):
    # Find the sequential_site corresponding to the reference_site
    reference_site_row = site_numbering_data[site_numbering_data['reference_site'] == site]
    if reference_site_row.empty:
        raise ValueError(f"Reference site {site} not found in site_numbering_map.csv")
    
    sequential_site = reference_site_row['sequential_site'].values[0]
    
    # Verify that the wildtype matches the sequence at the sequential_site position
    if sequence[sequential_site - 1] != wildtype:
        print(sequence[sequential_site - 1], wildtype)
        raise ValueError(f"Wildtype {wildtype} does not match the sequence at position {sequential_site}")

    # Replace the amino acid in the sequence at the sequential_site position with the mutant amino acid
    mutated_sequence = sequence[:sequential_site - 1] + mutant + sequence[sequential_site:]

    return mutated_sequence.replace("*", "G").replace("-", "G")

class HIVEscapeDataset(Dataset):
    def __init__(self, escape_data = './data/3BNC117_avg.csv'):
        super().__init__()
        self.escape_Data = read_csv_to_list(escape_data)

    def __len__(self):
        return len(self.escape_Data)

    def __getitem__(self, idx):# {spike, VH, VL, ic50 }
        res = self.escape_Data[idx]
        # RBD_start = res['spike'].find('GVP') - 8
        # RBD_end = max(res['spike'].find('APT')+15, res['spike'].find('YKVV')+25)
        # res['spike'] = res['spike'][RBD_start:RBD_end+1]
        return res


