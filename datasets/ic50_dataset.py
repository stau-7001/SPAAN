import os
from torch.utils.data import Dataset
import torch
import csv
import esm
from tqdm import tqdm
import torch.nn as nn
import pickle

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


def read_csv_to_list(file_path):
    result = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row

        for row in reader:
            vh_seq = row[-2]
            vl_seq = row[-1]
            for i in range(1, len(header) - 2):
                ic50 = row[i]
                if ">" in ic50:
                    ic50 = MAX_IC
                else:
                    try:
                        ic50 = min(MAX_IC, float(ic50))
                        if header[i] == "RBD" and ic50 != MAX_IC:
                            ic50 = min(MAX_IC, float(ic50) * 0.001)
                    except ValueError:
                        # Skip the data if ic50 is not a number
                        continue
                result.append(
                    {
                        "H": vh_seq,
                        "L": vl_seq,
                        "virus": header[i],
                        "IC50": ic50,
                        # 'H_CDR3':row[-4],
                        # 'L_CDR3':row[-1]
                    }
                )
    return result


class ESM_encoder(nn.Module):
    def __init__(self, device, **kwargs):
        super(ESM_encoder, self).__init__()
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, data):
        data = [("", data[i]) for i in range(len(data))]
        _, _, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[30], return_contacts=True)
        return (
            results["representations"][30],
            batch_lens,
        )  # shape: (batch_size, max_seq_len, esm_hidden_dim=640), (batch_size,)


class IC50Dataset(Dataset):
    def __init__(
        self,
        spike_data="./data/Ag_sequence.csv",
        ic50_data="./data/updated_processed_data.csv",
    ):
        super().__init__()
        self.ic_50_data = read_csv_to_list(ic50_data)
        self.spike_dict = read_csv_to_dict(spike_data)
        self.preprocess_data()
        self.preprocess_virus()

    def __len__(self):
        return len(self.ic_50_data)

    # def preprocess_data(
    #     self, save_path="./data/preprossed/precomputed_embeddings_sarscov_ic50.pth"
    # ):
    #     if os.path.exists(save_path):
    #         self.precomputed_embeddings = torch.load(save_path)
    #         return

    #     print("Precomputing ESM embedding...")
    #     esm_encoder = ESM_encoder(
    #         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     )
    #     all_H_embeddings = []
    #     all_H_lens = []
    #     all_L_embeddings = []
    #     all_L_lens = []
    #     all_ag_embeddings = []
    #     all_ag_lens = []

    #     # 使用 tqdm 显示进度条
    #     for entry in tqdm(self.ic_50_data, desc="Processing ESM Embeddings"):
    #         data_H = entry["H"]
    #         data_L = entry["L"]
    #         data_ag, _ = self.spike_dict[entry["virus"]]

    #         # 单个计算
    #         esm_encoder.model.eval()
    #         with torch.no_grad():
    #             H_embeddings, H_lens = esm_encoder([data_H])
    #             L_embeddings, L_lens = esm_encoder([data_L])
    #             ag_embeddings, ag_lens = esm_encoder([data_ag])
    #         all_H_embeddings.append(H_embeddings[0].cpu())
    #         all_L_embeddings.append(L_embeddings[0].cpu())
    #         all_ag_embeddings.append(ag_embeddings[0].cpu())
    #         all_H_lens.append(H_lens[0].item())
    #         all_L_lens.append(L_lens[0].item())
    #         all_ag_lens.append(ag_lens[0].item())

    #     # 将所有计算结果保存
    #     self.precomputed_embeddings = {
    #         "H_embeddings": all_H_embeddings,  # List of tensors
    #         "H_lens": all_H_lens,  # List of lengths
    #         "L_embeddings": all_L_embeddings,  # List of tensors
    #         "L_lens": all_L_lens,  # List of lengths
    #         "ag_embeddings": all_ag_embeddings,  # List of tensors
    #         "ag_lens": all_ag_lens,  # List of lengths
    #     }

    #     # 保存到文件
    #     torch.save(self.precomputed_embeddings, save_path)
    #     print(f"预计算的 embedding 已保存到 {save_path}")
    def preprocess_virus(
        self,
        save_file="./data/preprocessed/preprocessed_sars_ic50/spikes.pkl",
    ):
        if os.path.exists(save_file):
            print(f"spike file {save_file} already exists. Skipping preprocessing.")
            with open(save_file, "rb") as pkl_file:
                self.spike_embs = pickle.load(pkl_file)
            return
        self.spike_embs = {}
        esm_encoder = ESM_encoder(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        for virus, value in self.spike_dict.items():
            # print(virus, seq)
            seq, _ = value
            esm_encoder.model.eval()
            with torch.no_grad():
                ag_embeddings, ag_lens = esm_encoder([seq])

            self.spike_embs[virus] = (ag_embeddings[0].cpu(), ag_lens[0].item())
        with open(save_file, "wb") as pkl_file:
            pickle.dump(self.spike_embs, pkl_file)

    def preprocess_data(
        self,
        save_dir="./data/preprocessed/preprocessed_sars_ic50/",
        csv_file="./data/preprocessed/preprocessed_sars_ic50/metadata.csv",
        old_pth_file="./data/preprossed/precomputed_embeddings_sarscov_ic50.pth",
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
                data_H = entry["H"]
                data_L = entry["L"]
                data_ag, _ = self.spike_dict[entry["virus"]]

                # Compute embeddings
                esm_encoder.model.eval()
                with torch.no_grad():
                    H_embeddings, H_lens = esm_encoder([data_H])
                    L_embeddings, L_lens = esm_encoder([data_L])
                    ag_embeddings, ag_lens = esm_encoder([data_ag])

                # Organize data
                precomputed_data = {
                    "H_embeddings": H_embeddings[0].cpu(),
                    "H_lens": H_lens[0].item(),
                    "L_embeddings": L_embeddings[0].cpu(),
                    "L_lens": L_lens[0].item(),
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
                        precomputed_data["H_lens"],
                        precomputed_data["L_lens"],
                        precomputed_data["ag_lens"],
                    ]
                )

        print(
            f"Precomputed embeddings and metadata saved to {save_dir} and {csv_file}."
        )

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

    def __getitem__(self, idx, use_rbd=False, use_esm=False):
        res = self.ic_50_data[idx]
        res["spike"], res["weight"] = self.spike_dict[res["virus"]]
        if use_rbd:
            RBD_start = res["spike"].find("NITN")
            RBD_end = res["spike"].find("KKST") + 4
            res["spike"] = res["spike"][RBD_start : RBD_end + 1]

        # 加入预计算的 H 和 L embedding 及其长度
        pkl_filename = f"/gpfs/share/home/2201111701/lyt/SPAAN_release/data/preprocessed/preprocessed_sars_ic50/entry_{idx}.pkl"
        # abs_pkl_filename = os.path.join(os.path.dirname(__file__), pkl_filename)
        Ag_embedding, Ag_length = self.spike_embs[res["virus"]]
        if use_esm:
            if os.path.exists(pkl_filename):
                with open(pkl_filename, "rb") as pkl_file:
                    precomputed_data = pickle.load(pkl_file)
                    pkl_file.close()
                    res.update(
                        {
                            "H_embedding": precomputed_data["H_embeddings"],
                            "H_length": precomputed_data["H_lens"],
                            "L_embedding": precomputed_data["L_embeddings"],
                            "L_length": precomputed_data["L_lens"],
                            "Ag_embedding": Ag_embedding,
                            "Ag_length": Ag_length,
                            # "Ag_embedding": precomputed_data["ag_embeddings"],
                            # "Ag_length": precomputed_data["ag_lens"]
                        }
                    )
                    del precomputed_data
            else:
                raise FileNotFoundError(
                    f"Precomputed data file {pkl_filename} not found."
                )
        else:
            res.update(
                {
                    "H_embedding": None,
                    "H_length": None,
                    "L_embedding": None,
                    "L_length": None,
                    "Ag_embedding": Ag_embedding,
                    "Ag_length": Ag_length,
                    # "Ag_embedding": None,
                    # "Ag_embedding": precomputed_data["ag_embeddings"],
                    # "Ag_length": precomputed_data["ag_lens"]
                }
            )
        return res


def count_similar_entries(dataset, ic50_threshold=0.5, residue_diff_threshold=5):
    count = 0
    negetive_count = 0
    positive_count = 0

    for i, entry_1 in enumerate(dataset.ic_50_data):
        if entry_1["IC50"] >= 10.0:
            negetive_count += 1
        else:
            positive_count += 1
        # for j, entry_2 in enumerate(dataset.ic_50_data):
        #     # Skip self-comparison
        #     if i == j:
        #         continue

        #     # Calculate the Levenshtein distance between antibody sequences (H or L chains)
        #     h_diff = distance(entry_1["H"], entry_2["H"])
        #     l_diff = distance(entry_1["L"], entry_2["L"])

        #     # Check if H or L sequence difference is <= 5 residues
        #     if h_diff <= residue_diff_threshold or l_diff <= residue_diff_threshold:
        #         # Check if IC50 difference is greater than the threshold
        #         ic50_diff = abs(entry_1["ic50"] - entry_2["ic50"])
        #         if ic50_diff > ic50_threshold:
        #             count += 1

    print(f"Negative count: {negetive_count}")
    print(f"Positive count: {positive_count}")
    return count


if __name__ == "__main__":
    dataset = IC50Dataset()
    count_similar_entries(dataset)
    print(f"Number of entries: {len(dataset)}")
