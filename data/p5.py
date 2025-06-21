"""
Script to process the dataset in P5 way for Sequential Recommendation.
"""

# imports
from collections import defaultdict
import os
import sys
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
import datasets
from datasets import load_dataset
import argparse
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from modules.utils import display_args, set_seed
datasets.logging.set_verbosity_error()
import logging

# create logger
logger = logging.getLogger("recsys_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = RichHandler(show_path=False)
    logger.addHandler(handler)
    logger.propagate = False
    

# global amazon dataset mappings
DATA_NAME_MAP = {
    'beauty': 'All_Beauty',
    'toys': 'Toys_and_Games',
    'sports': 'Sports_and_Outdoors',
    'games': 'Video_Games',
    'software': "Software",
}
INVERSE_DATA_NAME_MAP = {v: k for k, v in DATA_NAME_MAP.items()}


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def parse_2023(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Parsing {path}"):
            yield json.loads(line)


def display_pickle_summary(data, title="Pickle File Contents"):
    """
    Load and summarize the contents of a pickle file using rich.

    :param data: .pkl data
    :param title: Optional title for the printed table.
    """
    table = Table(title=title)

    table.add_column("Key/Type", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")

    total_size = 0
    if isinstance(data, dict):
        for key, value in data.items():
            desc = f"{type(value).__name__}, len={len(value)}" if hasattr(
                value, '__len__') else type(value).__name__
            table.add_row(str(key), desc)
            if key in ["train", "test", "val"]:
                total_size += len(value)
            else:
                total_size = "N/A"
    else:
        table.add_row(type(data).__name__, f"{data}" if isinstance(
            data, (int, float, str)) else str(type(data)))

    table.add_row("Total Size", str(total_size))
    console = Console()
    console.print(table)

    if "train" in data or "test" in data or "val" in data:
        rprint("Train Sample:")
        rprint(data['train'][0])
        rprint("Val Sample:")
        rprint(data['val'][0])
        rprint("Test Sample:")
        print(data['test'][0])


def df_stats(df: pd.DataFrame, title="DataFrame Stats"):
    table = Table(title=title)
    rprint(f"DataFrame shape: {df.shape}")
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Non-Null Count", style="yellow")
    table.add_column("Unique Count", style="magenta")
    table.add_column("Null/NA Count", style="red")
    table.add_column("Data Type", style="green")

    for col in df.columns:
        try:
            non_null_count = df[col].notna().sum()
        except:
            non_null_count = "Error"
        try:
            unique_count = df[col].nunique(dropna=True)
        except:
            unique_count = "Error"
        try:
            null_count = df[col].isna().sum()
        except:
            null_count = "Error"
        try:
            dtype = str(df[col].dtype)
        except:
            dtype = "Error"
        table.add_row(col, str(non_null_count), str(
            unique_count), str(null_count), dtype)

    Console().print(table)


def filter_amazon_dataset(category, metadata=False):
    logger.info(f"Loading `{category}` {'Metadata' if metadata else 'Reviews'}")
    raw_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                               f"raw_meta_{category}" if metadata else f"raw_review_{category}",
                               trust_remote_code=True)
    raw_df = raw_dataset['full'].to_pandas()
    # df_stats(raw_df, f"{category} Reviews DataFrame Stats")

    logger.info(f"Loading `{category}` Ratings")
    rating_file = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename=f'benchmark/5core/rating_only/{category}.csv',
        repo_type='dataset'
    )
    rating_df = pd.read_csv(rating_file)
    # df_stats(rating_df, f"{category} Rating DataFrame Stats")

    # create sets for filtering
    valid_users = set(rating_df['user_id'].unique())
    valid_items = set(rating_df['parent_asin'].unique())

    logger.info(f"Filtering `{category}` Data")
    # filter reviews where both user_id and parent_asin are in the 5-core subset
    if metadata:
        filtered_data = raw_df[raw_df['parent_asin'].isin(valid_items)]
    else:
        filtered_data = raw_df[
            (raw_df['user_id'].isin(valid_users)) &
            (raw_df['parent_asin'].isin(valid_items))
        ]

    # df_stats(filtered_reviews, f"{category} Filtered Reviews DataFrame Stats")
    # rprint("Filtered Dataset Shape:", filtered_data.shape)

    return filtered_data


def get_reviews(dataset_dir, dataset_name, rating_score, year=2023):
    '''
    2014-reviews: https://jmcauley.ucsd.edu/data/amazon/index_2014.html
    2023-reviews: https://amazon-reviews-2023.github.io/main.html#for-user-reviews
    '''
    data = []
    # remove those with less than a certain score
    # older Amazon
    if not year == 2023:
        data_file = os.path.join(
            dataset_dir, f"reviews_{dataset_name}.json.gz")
        for review in parse(data_file):
            if float(review['overall']) <= rating_score:
                continue
            user = review['reviewerID']
            item = review['asin']
            time = review['unixReviewTime']
            data.append((user, item, int(time)))
    # latest Amazon
    else:
        # slow to parse directly, load with HF
        data_file = os.path.join(
            dataset_dir, INVERSE_DATA_NAME_MAP[dataset_name], "reviews.json.gz")
        if not os.path.exists(data_file):
            # raise FileExistsError("Save the K-Core file first!")
            # save the k-core first
            logger.info(f"Saving k-core reviews for `{dataset_name}`")
            filtered_reviews = filter_amazon_dataset(dataset_name)
            filtered_reviews.to_json(data_file,
                                     orient='records',
                                     lines=True,
                                     compression='gzip')
        for review in parse_2023(data_file):
            if float(review['rating']) <= rating_score:
                continue
            user = review['user_id']
            item = review['parent_asin']
            time = review['timestamp']
            data.append((user, item, int(time)))

    return data


def get_metadata(dataset_dir, dataset_name, data_maps, year=2023):
    '''
    2014-metadata: https://jmcauley.ucsd.edu/data/amazon/index_2014.html
    2023-metadata: https://amazon-reviews-2023.github.io/main.html#for-item-metadata
    '''
    data = {}
    item_asins = list(data_maps['item2id'].keys())

    # older Amazon
    if not year == 2023:
        meta_file = os.path.join(dataset_dir, f"meta_{dataset_name}.json.gz")
        for info in parse(meta_file):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
    # latest Amazon
    else:
        # slow to parse directly, load with HF
        meta_file = os.path.join(
            dataset_dir, INVERSE_DATA_NAME_MAP[dataset_name], "meta.json.gz")
        if not os.path.exists(meta_file):
            # raise FileExistsError("Save the K-Core file first!")
            logger.info(f"Saving k-core metadata for `{dataset_name}`")
            # save the k-core first
            filtered_meta = filter_amazon_dataset(dataset_name, metadata=True)
            filtered_meta.to_json(meta_file,
                                  orient='records',
                                  lines=True,
                                  compression='gzip')
        for info in parse_2023(meta_file):
            # comparsion not required as filtered data is already correct
            # also, super slow parsing! need a better approach
            # if info['parent_asin'] not in item_asins:
            #     continue
            data[info['parent_asin']] = info
    return data


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]


def get_attribute_amazon(meta_infos, datamaps, attribute_core, year=2023):
    attributes = defaultdict(int)
    for iid, info in tqdm(meta_infos.items()):
        for cates in info['categories']:
            for cate in cates[1:]:
                attributes[cate] += 1
        if year == 2023:
            brand = eval(info["details"]).get("Brand", "Unknown")
            attributes[brand] += 1
        else:
            attributes[info['brand']] += 1

    logger.info(f'Before Delete, Attribute Num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm(meta_infos.items()):
        new_meta[iid] = []

        if year == 2023:
            brand = eval(info["details"]).get("Brand", "Unknown")
            if attributes[brand] >= attribute_core:
                new_meta[iid].append(brand)
        else:
            if attributes[info['brand']] >= attribute_core:
                new_meta[iid].append(info['brand'])

        for cates in info['categories']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    logger.info(f'After delete, Attribute Num:{len(attribute2id)}')
    logger.info(f'Attributes Length, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')

    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num

    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq


def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True  # 已经保证Kcore


def filter_Kcore(user_items, user_core, item_core):  # user 接所有items
    user_count, item_count, isKcore = check_Kcore(
        user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(
            user_items, user_core, item_core)
    return user_items


def id_map_old(user_items):  # user_items dict
    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid
    id2user = {}  # uid 2 raw
    id2item = {}  # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []  # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def id_map_new(user_items): # user_items dict
    """
    From the TIGER author
    """
    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 0
    item_id = 0
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    
    user_set = set()
    item_set = set()
    for user in random_user_list:
        user_set.add(user)
        items = user_items[user]
        item_set.update(items)
        
    random_user_mapping = [str(i+1) for i in range(len(user_set))]
    random_item_mapping = [str(i+1) for i in range(len(item_set))]
    random.shuffle(random_user_mapping)
    random.shuffle(random_item_mapping)
    
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(random_user_mapping[user_id])
            id2user[str(random_user_mapping[user_id])] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(random_item_mapping[item_id])
                id2item[str(random_item_mapping[item_id])] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id, item_id, data_maps


def main(dataset_dir, data_name, acronym, data_type='Amazon', new_session_creator=True):
    assert data_type in {'Amazon', 'Yelp'}
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    if data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        review_data = Yelp(date_min, date_max, rating_score)
    else:
        review_data = get_reviews(dataset_dir, data_name, rating_score)

    user_items = get_interaction(review_data)
    logger.info(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')

    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(
        user_items, user_core=user_core, item_core=item_core)
    logger.info(f'User {user_core}-core complete! Item {item_core}-core complete!')

    if new_session_creator:
        user_items, user_num, item_num, data_maps = id_map_new(user_items)
    else:
        user_items, user_num, item_num, data_maps = id_map_old(user_items)
    user_count, item_count, _ = check_Kcore(
        user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(
        user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(
        item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    logger.info(show_info)

    logger.info('Extracting Meta Information')

    if data_type == 'Amazon':
        meta_infos = get_metadata(dataset_dir, data_name, data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_amazon(
            meta_infos, data_maps, attribute_core)
    else:
        meta_infos = Yelp_meta(data_maps)  # TODO: add yelp integration
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(
            meta_infos, data_maps, attribute_core)
    
    rprint(
        f'{data_name} | Users: {add_comma(user_num)} | Items: {add_comma(item_num)} | '
        f'UserAvg: {user_avg:.1f} | ItemAvg: {item_avg:.1f} | '
        f'Interactions: {add_comma(interact_num)} | Sparsity: {sparsity:.2f}% | '
        f'Attributes: {add_comma(attribute_num)} | AttrAvg: {avg_attribute:.1f} \\'
    )

    # -------------- Save Data ---------------
    seq_data_file = os.path.join(dataset_dir, acronym, 'sequential_data.txt')
    item2attributes_file = os.path.join(
        dataset_dir, acronym, 'item2attributes.json')
    datamaps_file = os.path.join(dataset_dir, acronym, 'datamaps.json')

    with open(seq_data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)

    json_str = json.dumps(datamaps)
    with open(datamaps_file, 'w') as out:
        out.write(json_str)


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Download and pre-p5-process the dataset.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="../dataset/amazon/2023/raw",
                        help='Directory containing the dataset')
    parser.add_argument('--dataset_split', type=str,
                        default="beauty", help='Dataset split to process')
    parser.add_argument('--data_type', type=str,
                        default="Amazon", help='Parent Dataset Type')
    parser.add_argument('--new_session_creator', type=bool,
                        default=True, help='Old or New Session Creator')
    args = parser.parse_args()
    display_args(args)

    # set seed
    set_seed(seed=2025)

    data_type = args.data_type
    category = args.dataset_split
    dataset_dir = args.dataset_dir
    new_session_creator = args.new_session_creator
    os.makedirs(os.path.join(dataset_dir, category), exist_ok=True)

    # start processing
    main(dataset_dir, DATA_NAME_MAP[category], category, data_type, new_session_creator)
