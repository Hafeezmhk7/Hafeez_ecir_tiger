import gzip
from collections import defaultdict
from datetime import datetime
from os.path import join as opj

DATA_DIR = 'data/'
def parse(path):
    # In Python 3, gzip.open in read mode ('r') without 't' (text) opens in binary mode.
    # To read lines as strings, use 'rt' and specify encoding.
    g = gzip.open(path, 'rt', encoding='utf-8')
    for l in g:
        yield eval(l)

def preprocess(dataset_name='Sports_and_Outdoors'):
    """
    Preprocess the dataset by reading reviews from a JSON file, filtering users and products based on review counts,
    and writing the processed data to a text file. It also creates mappings for users and items, ensuring that
    only those with at least 5 reviews are included. The output is a text file where each line contains a user ID
    and an item ID, representing a review interaction.
    """
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    # It's good practice to specify encoding when opening files for writing text.
    with open(opj(DATA_DIR, 'reviews_' + dataset_name + '.txt'), 'w', encoding='utf-8') as f:
        for l in parse(opj(DATA_DIR, 'reviews_' + dataset_name + '.json.gz')):
            line += 1
            f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
            asin = l['asin']
            rev = l['reviewerID']
            # time = l['unixReviewTime'] # This variable 'time' is assigned but not used in this loop
            countU[rev] += 1
            countP[asin] += 1

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()

    # Re-initialize line counter or use a different one if needed,
    # otherwise it continues from the previous loop.
    # Assuming line count is not critical for this second pass or should be reset.
    # If the line count was meant to be cumulative across both reads, it's fine.
    # For clarity, if it's a new count, it should be reset: line = 0

    for l in parse(opj(DATA_DIR, 'reviews_' + dataset_name + '.json.gz')):
        # line += 1 # If this line count is for the second pass, uncomment and potentially reset 'line' before loop
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])

    # sort reviews in User according to time
    for userid in User: # Iterating directly over dict keys is common in Python 3
        User[userid].sort(key=lambda x: x[0])

    # print statement becomes a function in Python 3
    print(usernum, itemnum)

    with open(opj(DATA_DIR, dataset_name + '.txt'), 'w', encoding='utf-8') as f:
        for user in User: # Iterating directly over dict keys
            for i in User[user]:
                f.write('%d %d\n' % (user, i[1]))

def get_stats(fname):
    num_users = 0
    num_items = 0
    User = defaultdict(list)
    # assume user/item index starting from 1
    f = open(opj(DATA_DIR, '%s.txt' % fname), 'r')

    total_actions = 0
    item_actions_count = defaultdict(int)

    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        num_users = max(u, num_users)
        num_items = max(i, num_items)
        User[u].append(i)
        total_actions += 1
        item_actions_count[i] += 1
    f.close()
    
    avg_actions_per_user = total_actions / num_users if num_users > 0 else 0
    avg_actions_per_item = total_actions / num_items if num_items > 0 else 0

    print(f"Statistics for {fname}:")
    print(f"  Number of users: {num_users}")
    print(f"  Number of items: {num_items}")
    print(f"  Number of actions: {total_actions}")
    print(f"  Avg actions per user: {avg_actions_per_user:.2f}")
    print(f"  Avg actions per item: {avg_actions_per_item:.2f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process dataset statistics.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset')
    parser.add_argument('--stats', action='store_true', help='Get statistics of the dataset')
    args = parser.parse_args()

    if args.preprocess:
        preprocess(args.dataset)
    if args.stats:
        get_stats(args.dataset)