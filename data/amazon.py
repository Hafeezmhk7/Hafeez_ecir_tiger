import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
from collections import defaultdict
try:
    from data.preprocessing import PreprocessingMixin
except:
    from preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import Optional, Dict, Union
import logging

# fetch logger
logger = logging.getLogger("recsys_logger")


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)

def parse_2023(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Parsing {path}"):
            yield json.loads(line)

class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        category="brand",
    ) -> None:
        self.split = split
        self.year = int(root.split("/")[-1]) # extract dataset year from the path
        self.brand_mapping = {}  # Dictionary to store brand_id -> brand_name mapping
        self.category = category
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        if not self.year == 2023:
            path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
            extract_zip(path, self.root)
            os.remove(path)
            folder = osp.join(self.root, "data")
            fs.rm(self.raw_dir)
            os.rename(folder, self.raw_dir)

    def _remap_ids(self, x):
        return x - 1

    def get_brand_name(self, brand_id: int) -> str:
        """
        Returns the brand name for a given brand ID.

        Args:
            brand_id: The ID of the brand to look up

        Returns:
            The brand name as a string, or "Unknown" if the brand ID is not found
        """
        return self.brand_mapping.get(brand_id, "Unknown")

    def get_brand_mapping(self) -> Dict[int, str]:
        """
        Returns the complete brand ID to brand name mapping.

        Returns:
            Dictionary mapping brand IDs to brand names
        """
        return self.brand_mapping

    def train_test_split(self, max_seq_len=20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        with open(
            os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r"
        ) as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]

                # We keep the whole sequence without padding. Allows flexible training-time subsampling.
                # example: items[22]
                train_items = items[:-2]  # items[0:20] → [1..20]
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2]) # → 21

                eval_items = items[-(max_seq_len + 2) : -2] # items[-22:-2] → [1..20]
                sequences["eval"]["itemId"].append(
                    eval_items + [-1] * (max_seq_len - len(eval_items))
                )
                sequences["eval"]["itemId_fut"].append(items[-2]) # → 21

                test_items = items[-(max_seq_len + 1) : -1] # items[-21:-1] → [2..21]
                sequences["test"]["itemId"].append(
                    test_items + [-1] * (max_seq_len - len(test_items))
                )
                sequences["test"]["itemId_fut"].append(items[-1]) # → 22

        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), "r") as f:
            data_maps = json.load(f)

        # Construct user sequences
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"]) for k, v in sequences.items()
        }

        # Compute item features
        asin2id = pd.DataFrame(
            [
                {"asin": k, "id": self._remap_ids(int(v))}
                for k, v in data_maps["item2id"].items()
            ]
        )
        
        if self.year == 2023:
            meta_df =  pd.DataFrame(
                [
                    meta
                    for meta in parse_2023(
                        path=os.path.join(self.raw_dir, self.split, "meta.json.gz")
                    )
                ]
            )
            # process meta df
            meta_df.rename(columns={"parent_asin": "asin"}, inplace=True)
            meta_df["brand"] = meta_df["details"].apply(lambda x: eval(x).get("Brand", "Unknown"))
            item_data = (meta_df
                .merge(asin2id, on="asin")
                .sort_values(by="id")
                .fillna({"brand": "Unknown"})
            )
        else:
            item_data = (
                pd.DataFrame(
                    [
                        meta
                        for meta in parse(
                            path=os.path.join(self.raw_dir, self.split, "meta.json.gz")
                        )
                    ]
                )
                .merge(asin2id, on="asin")
                .sort_values(by="id")
                .fillna({"brand": "Unknown"})
            )

        # save item data
        raw_folder = osp.join(self.root, "raw", self.split)
        item_data.to_csv(os.path.join(raw_folder, "item_data.csv"), index=False)
        
        # Create brand mapping
        unique_brands = item_data[self.category].unique()
        self.brand_mapping = {i: brand for i, brand in enumerate(unique_brands)}

        # Create reverse mapping for lookup
        brand_to_id = {brand: i for i, brand in self.brand_mapping.items()}

        # Add brand_id to item_data
        item_data["brand_id"] = item_data["brand"].map(lambda x: brand_to_id.get(x, -1))

        if self.year == 2023:
            sentences = item_data.apply(
                lambda row: "Title: "
                + str(row["title"])
                + "; "
                + "Brand: "
                + str(row["brand"])
                + "; "
                + "Categories: "
                + (str(row["categories"]) if row["categories"] else f'[{row["main_category"]}]')
                + "; "
                + "Rating: "
                + str(row["average_rating"])
                + "; "
                + "Price: "
                + str(row["price"])
                + "; ",
                axis=1,
            )
        else:
            sentences = item_data.apply(
                lambda row: "Title: "
                + str(row["title"])
                + "; "
                + "Brand: "
                + str(row["brand"])
                + "; "
                + "Categories: "
                + str(row["categories"][0])
                + "; "
                + "Price: "
                + str(row["price"])
                + "; ",
                axis=1,
            )

        # Store brand_id instead of brand name
        brand_ids = item_data.apply(lambda row: row["brand_id"], axis=1)

        item_emb = self._encode_text_feature(sentences)
        data["item"].x = item_emb
        data["item"].text = np.array(sentences)
        data["item"].brand_id = np.array(
            brand_ids
        )  # Store brand_id instead of brand name

        # Save the brand mapping to the data object as well
        data["brand_mapping"] = self.brand_mapping

        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        self.save([data], self.processed_paths[0])

        # Save brand mapping to a separate file for easy access
        brand_mapping_path = os.path.join(
            self.processed_dir, f"brand_mapping_{self.split}.json"
        )
        with open(brand_mapping_path, "w") as f:
            json.dump(self.brand_mapping, f)


if __name__ == "__main__":
    root = "dataset/amazon/2023"
    split = "beauty"
    AmazonReviews(root, split)