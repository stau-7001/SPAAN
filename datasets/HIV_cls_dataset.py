import os
from torch.utils.data import Dataset
import torch
import csv

MAX_IC = 10.0
from datasets.ic50_dataset import ESM_encoder
import pickle


def parse_weight_vector(weight_vector_str):
    return (
        [1.0] + [float(weight) for weight in weight_vector_str[1:-1].split(",")] + [1.0]
    )


def read_csv_to_dict(file_path):
    result = {}
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            col_1, col_2, col_3 = row[0], row[2], parse_weight_vector(row[3])
            result[col_1] = (col_2, torch.tensor(col_3))
    return result


class HIVClsDataset(Dataset):
    def __init__(self, data_file="./data/HIV/unseen/train_seed_0.csv"):
        super().__init__()
        self.ic_50_data = self.read_csv_to_list(data_file)
        if not os.path.exists("./data/preprocessed/preprocessed_HIV"):
            print("preprocessing")
            os.makedirs("./data/preprocessed/preprocessed_HIV")
            self.preprocess_ab()
            self.preprocess_ag()
        self.ab_list, self.ag_list = [], []
        idx = 0
        while True:
            file_path = f"/gpfs/share/home/2201111701/lyt/SPAAN_release/data/preprocessed/preprocessed_HIV/abentry_{idx}.pkl"
            if not os.path.exists(file_path):
                break
            with open(file_path, "rb") as f:
                self.ab_list.append(pickle.load(f))
            idx += 1

        idx = 0
        while True:
            file_path = f"/gpfs/share/home/2201111701/lyt/SPAAN_release/data/preprocessed/preprocessed_HIV/agentry_{idx}.pkl"
            if not os.path.exists(file_path):
                break
            with open(file_path, "rb") as f:
                self.ag_list.append(pickle.load(f))
            idx += 1

    def __len__(self):
        return len(self.ic_50_data)

    def read_csv_to_list(self, file_path):
        result = []
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    cls_label = float(row["label"])
                except ValueError:
                    continue
                result.append(
                    {
                        "ab_idx": row["ab_id"],
                        "ag_idx": row["ag_id"],
                        "label": cls_label,
                    }
                )
        return result

    def preprocess_ab(
        self,
        save_dir="./data/preprocessed/preprocessed_HIV/",
        csv_path="./data/HIV/antibody.csv",
    ):
        esm_encoder = ESM_encoder(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # spike_dict = {}
        # for res in self.ic_50_data:
        #     spike_dict{res['site']+res['mutation']} = res['spike']

        with open(csv_path, "r", encoding="utf-8") as file:
            # print(virus, seq)
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                data_H, data_L = row["heavy"], row["light"]
                esm_encoder.model.eval()
                with torch.no_grad():
                    # ag_embeddings, ag_lens = esm_encoder([seq])
                    H_embeddings, H_lens = esm_encoder([data_H])
                    L_embeddings, L_lens = esm_encoder([data_L])
                tmp = {}
                tmp["H"] = (H_embeddings[0].cpu(), H_lens[0].item(), data_H)
                tmp["L"] = (L_embeddings[0].cpu(), L_lens[0].item(), data_L)
                save_file = os.path.join(save_dir, f"abentry_{idx}.pkl")
                with open(save_file, "wb") as pkl_file:
                    pickle.dump(tmp, pkl_file)

    def preprocess_ag(
        self,
        save_dir="./data/preprocessed/preprocessed_HIV/",
        csv_path="./data/HIV/antigen.csv",
    ):
        esm_encoder = ESM_encoder(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # spike_dict = {}
        # for res in self.ic_50_data:
        #     spike_dict{res['site']+res['mutation']} = res['spike']

        with open(csv_path, "r", encoding="utf-8") as file:
            # print(virus, seq)
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                seq = row["ag_seq"]
                esm_encoder.model.eval()
                with torch.no_grad():
                    ag_embeddings, ag_lens = esm_encoder([seq])
                ag_embs = (ag_embeddings[0].cpu(), ag_lens[0].item(), seq)
                save_file = os.path.join(save_dir, f"agentry_{idx}.pkl")
                with open(save_file, "wb") as pkl_file:
                    pickle.dump(ag_embs, pkl_file)

    def __getitem__(self, idx):  # {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        max_l = 140
        max_h = 140
        res["H_embedding"], res["H_length"], res["H"] = self.ab_list[
            int(res["ab_idx"])
        ]["H"]
        # print(res["H_embedding"].shape, res["H_length"], len(res["H"]))

        res["H_embedding"] = res["H_embedding"][: max_h + 2]
        res["H"] = res["H"][:max_h]
        res["H_length"] = min(res["H_length"], max_h + 2)

        res["L_embedding"], res["L_length"], res["L"] = self.ab_list[
            int(res["ab_idx"])
        ]["L"]

        res["L_embedding"] = res["L_embedding"][: max_l + 2]
        res["L"] = res["L"][:max_l]
        res["L_length"] = min(res["L_length"], max_l + 2)

        res["Ag_embedding"], res["Ag_length"], res["Ag_seq"] = self.ag_list[
            int(res["ag_idx"])
        ]
        # abpkl_filename = f"/gpfs/share/home/2201111701/lyt/SPAAN_release/data/preprocessed/preprocessed_HIV/abentry_{res['ab_idx']}.pkl"
        # # abs_pkl_filename = os.path.join(os.path.dirname(__file__), pkl_filename)
        # if os.path.exists(abpkl_filename):
        #     with open(abpkl_filename, "rb") as pkl_file:
        #         precomputed_data = pickle.load(pkl_file)
        #         pkl_file.close()
        #     res["H_embedding"],res["H_length"],res["H"] = precomputed_data["H"]
        #     res["L_embedding"],res["L_length"],res["L"]= precomputed_data["L"]
        # else:
        #     raise FileNotFoundError(f"Precomputed data file {abpkl_filename} not found.")

        # agpkl_filename = f"/gpfs/share/home/2201111701/lyt/SPAAN_release/data/preprocessed/preprocessed_HIV/agentry_{res['ag_idx']}.pkl"
        # if os.path.exists(agpkl_filename):
        #     with open(agpkl_filename, "rb") as pkl_file:
        #         precomputed_data = pickle.load(pkl_file)
        #         res["Ag_embedding"],res["Ag_length"],res["Ag_seq"] = precomputed_data
        # else:
        #     raise FileNotFoundError(f"Precomputed data file {agpkl_filename} not found.")

        return res


# class HIVClsDataset(Dataset):
#     def __init__(self, n01_data = './data/dataset_hiv_cls.csv'):
#         super().__init__()
#         self.ic_50_data = read_csv_to_list(n01_data)

#     def __len__(self):
#         return len(self.ic_50_data)

#     def __getitem__(self, idx):# {spike, VH, VL, ic50 }
#         res = self.ic_50_data[idx]
#         # RBD_start = res['spike'].find('GVP') - 8
#         # RBD_end = max(res['spike'].find('APT')+15, res['spike'].find('YKVV')+25)
#         # res['spike'] = res['spike'][RBD_start:RBD_end+1]
#         return res
