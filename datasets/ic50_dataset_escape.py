import os
from torch.utils.data import Dataset
import torch
import csv
from datasets.ic50_dataset import ESM_encoder
import pickle
from tqdm import tqdm

MAX_IC = 10.0


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


def read_antibody_csv_to_dict(file_path):
    result = {}
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            name, H_chain, L_chain = row[0], row[-2], row[-1]
            result[name] = (H_chain, L_chain)
    return result


def read_csv_to_list(file_path, seq):
    # antibody,site,mutation,mut_escape,group
    # BD-196,349,H,0.2088664262572558,C
    result = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row
        for i, row in enumerate(reader):
            escape_score = float(row[3])

            # if epitope == 'E3':
            spike = mutate(seq, int(row[1]), row[2])

            result.append(
                {
                    "antibody_name": row[0],
                    "virus": "RBD",
                    "IC50": escape_score,
                    "site": row[1],
                    "mutation": row[2],
                    "group": row[-1],
                    "spike": spike,
                }
            )
    return result


def mutate(seq, site, mutation):
    # print(site,mutation)
    site -= 1
    seq = seq[:site] + mutation + seq[site + 1 :]  # 在指定 site 上进行氨基酸突变
    return seq


class IC50DatasetEscape(Dataset):
    def __init__(
        self,
        spike_data="./data/Ag_sequence.csv",
        antibody_data="./data/antibody_info.csv",
        ic50_data="./data/use_res_clean.csv",
    ):
        super().__init__()
        self.spike_dict = read_csv_to_dict(spike_data)
        seq, _ = self.spike_dict["RBD"]
        self.ic_50_data = read_csv_to_list(ic50_data, seq)
        self.antibody_dict = read_antibody_csv_to_dict(antibody_data)
        if not os.path.exists("./data/preprocessed/preprocessed_escape"):
            self.preprocess_ab()
            self.preprocess_ag()

    def __len__(self):
        return len(self.ic_50_data)

    def preprocess_data(
        self,
        save_dir="./data/preprocessed/preprocessed_escape/",
        csv_file="./data/preprocessed/preprocessed_escape/metadata.csv",
        old_pth_file="./data/preprossed/precomputed_embeddings_escape.pth",
    ):
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(csv_file):
            print(f"CSV file {csv_file} already exists. Skipping preprocessing.")
            return

        if os.path.exists(old_pth_file):
            print(f"Converting old .pth file {old_pth_file} to new format...")
            self.convert_pth_to_new_format(old_pth_file, save_dir, csv_file)
            return

        print("Precomputing ESM embedding...")
        esm_encoder = ESM_encoder(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Prepare CSV file
        with open(csv_file, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["pkl_file", "H_lens", "L_lens", "virus"])

            for idx, entry in enumerate(
                tqdm(self.ic_50_data, desc="Processing ESM Embeddings")
            ):
                data_H, data_L = self.antibody_dict[entry["antibody_name"]]
                data_ag = entry["spike"]

                # Compute embeddings
                esm_encoder.model.eval()
                with torch.no_grad():
                    # H_embeddings, H_lens = esm_encoder([data_H])
                    # L_embeddings, L_lens = esm_encoder([data_L])
                    ag_embeddings, ag_lens = esm_encoder([data_ag])

                # Organize data
                precomputed_data = {
                    # "H_embeddings": H_embeddings[0].cpu(),
                    # "H_lens": H_lens[0].item(),
                    # "L_embeddings": L_embeddings[0].cpu(),
                    # "L_lens": L_lens[0].item(),
                    "ag_embeddings": ag_embeddings[0].cpu(),
                    "ag_lens": ag_lens[0].item(),
                    # "virus": entry["virus"],
                }

                # Save as individual .pkl file
                pkl_filename = os.path.join(save_dir, f"entry_{idx}.pkl")
                with open(pkl_filename, "wb") as pkl_file:
                    pickle.dump(precomputed_data, pkl_file)

                # Write metadata to CSV
                csv_writer.writerow(
                    [
                        pkl_filename,
                        # precomputed_data["H_lens"],
                        # precomputed_data["L_lens"],
                        precomputed_data["ag_lens"],
                    ]
                )

        print(
            f"Precomputed embeddings and metadata saved to {save_dir} and {csv_file}."
        )

    def preprocess_ag(
        self,
        save_dir="./data/preprocessed/preprocessed_escape/",
    ):
        esm_encoder = ESM_encoder(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        spike_dict = {}
        for res in self.ic_50_data:
            name = res["site"] + res["mutation"]
            spike_dict[name] = res["spike"]

        for virus, seq in spike_dict.items():
            # print(virus, seq)
            esm_encoder.model.eval()
            with torch.no_grad():
                ag_embeddings, ag_lens = esm_encoder([seq])
            ag_embs = (ag_embeddings[0].cpu(), ag_lens[0].item())
            save_file = os.path.join(save_dir, f"agentry_{virus}.pkl")
            with open(save_file, "wb") as pkl_file:
                pickle.dump(ag_embs, pkl_file)

    def preprocess_ab(
        self,
        save_dir="./data/preprocessed/preprocessed_escape/",
    ):
        esm_encoder = ESM_encoder(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # spike_dict = {}
        # for res in self.ic_50_data:
        #     spike_dict{res['site']+res['mutation']} = res['spike']

        for name, seq in self.antibody_dict.items():
            # print(virus, seq)
            data_H, data_L = seq
            esm_encoder.model.eval()
            with torch.no_grad():
                # ag_embeddings, ag_lens = esm_encoder([seq])
                H_embeddings, H_lens = esm_encoder([data_H])
                L_embeddings, L_lens = esm_encoder([data_L])
            tmp = {}
            tmp["H"] = (H_embeddings[0].cpu(), H_lens[0].item())
            tmp["L"] = (L_embeddings[0].cpu(), L_lens[0].item())
            save_file = os.path.join(save_dir, f"abentry_{name}.pkl")
            with open(save_file, "wb") as pkl_file:
                pickle.dump(tmp, pkl_file)

    def convert_pth_to_new_format(self, old_pth_file, save_dir, csv_file):
        precomputed_embeddings = torch.load(old_pth_file)

        with open(csv_file, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["pkl_file", "H_lens", "L_lens", "virus"])

            for idx, (H_emb, H_len, L_emb, L_len, ag_emb, ag_len) in enumerate(
                zip(
                    precomputed_embeddings["H_embeddings"],
                    precomputed_embeddings["H_lens"],
                    precomputed_embeddings["L_embeddings"],
                    precomputed_embeddings["L_lens"],
                    precomputed_embeddings["ag_embeddings"],
                    precomputed_embeddings["ag_lens"],
                    # precomputed_embeddings["virus"],
                )
            ):
                precomputed_data = {
                    "H_embeddings": H_emb,
                    "H_lens": H_len,
                    "L_embeddings": L_emb,
                    "L_lens": L_len,
                    "ag_embeddings": ag_emb,
                    "ag_lens": ag_len,
                    # "virus": virus,
                }

                # Save as individual .pkl file
                pkl_filename = os.path.join(save_dir, f"entry_{idx}.pkl")
                with open(pkl_filename, "wb") as pkl_file:
                    pickle.dump(precomputed_data, pkl_file)

                # Write metadata to CSV
                csv_writer.writerow([pkl_filename, H_len, L_len, ag_len])

        print(
            f"Converted .pth file {old_pth_file} to new format and saved to {save_dir} and {csv_file}."
        )

    def __getitem__(self, idx, use_rbd=False):  # {spike, VH, VL, ic50 }
        res = self.ic_50_data[idx]
        # print(res)
        # seq,res['weight'] = self.spike_dict['RBD']
        # RBD_start = res['spike'].find('NITN')
        # RBD_end = res['spike'].find('KKST') + 4
        # print(RBD_start,RBD_end)
        res["H"], res["L"] = self.antibody_dict[res["antibody_name"]]
        if use_rbd:
            RBD_start = res["spike"].find("NITN")
            RBD_end = res["spike"].find("KKST") + 4
            res["spike"] = res["spike"][RBD_start : RBD_end + 1]

        abpkl_filename = f"/gpfs/share/home/2201111701/lyt/SPAAN_release/data/preprocessed/preprocessed_escape/abentry_{res['antibody_name']}.pkl"
        # abs_pkl_filename = os.path.join(os.path.dirname(__file__), pkl_filename)
        if os.path.exists(abpkl_filename):
            with open(abpkl_filename, "rb") as pkl_file:
                precomputed_data = pickle.load(pkl_file)
                pkl_file.close()
            res["H_embedding"], res["H_length"] = precomputed_data["H"]
            res["L_embedding"], res["L_length"] = precomputed_data["L"]
            del precomputed_data
        else:
            raise FileNotFoundError(
                f"Precomputed data file {abpkl_filename} not found."
            )

        agpkl_filename = f"/gpfs/share/home/2201111701/lyt/SPAAN_release/data/preprocessed/preprocessed_escape/agentry_{res['site']+res['mutation']}.pkl"
        if os.path.exists(agpkl_filename):
            with open(agpkl_filename, "rb") as pkl_file:
                precomputed_data = pickle.load(pkl_file)
                res["Ag_embedding"], res["Ag_length"] = precomputed_data
                pkl_file.close()
            del precomputed_data
        else:
            raise FileNotFoundError(
                f"Precomputed data file {agpkl_filename} not found."
            )
        return res


if __name__ == "__main__":
    dataset = IC50DatasetEscape()
    print(dataset[0])
# TODO: test
# dataset = IC50DatasetEscape()
# data = dataset[0]
# print(data)
