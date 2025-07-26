import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from os.path import join as opj

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


## S3 Rec code
def evaluate_full_dataset(model, dataset, args, top_k_range):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = defaultdict(float)
    HT = defaultdict(float)
    valid_user = 0.0

    users = range(1, usernum + 1)  # Iterate over all users

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # Prepare sequence using valid and train data, similar to 'evaluate'
        if valid[u]:  # Check if valid[u] is not empty
            seq[idx] = valid[u][0]
            idx -= 1
        
        for i in reversed(train[u]):
            if idx < 0:  # Check if sequence array is already full
                break
            seq[idx] = i
            idx -= 1
        
        true_test_item = test[u][0]  # Ground truth item from the test set

        # Candidate items for ranking: all items from 1 to itemnum
        all_items_for_ranking = np.arange(1, itemnum + 1) # 1-indexed item IDs

        # Prepare inputs for model.predict
        # user_ids: (batch_size=1,)
        # log_seqs: (batch_size=1, maxlen)
        # item_indices: (itemnum,) - assuming model.predict handles this for a single user
        # or (batch_size=1, itemnum) if model strictly expects 2D item_indices
        user_id_np = np.array([u])
        seq_np = np.array([seq])

        # Get raw scores for all items for the current user and sequence
        # Assuming model.predict(user_ids, log_seqs, item_indices_1D_or_2D) returns scores (batch_size, num_items)
        raw_scores_batch = model.predict(user_id_np, seq_np, all_items_for_ranking)
        
        # Negate scores because original 'evaluate' does, lower score = better rank
        scores = -raw_scores_batch[0]  # Shape (itemnum,)

        # Filter out items already seen by the user in train/valid sets
        # Set their scores to infinity so they are ranked last.
        items_to_filter = set(train[u])
        if valid[u]:
            items_to_filter.add(valid[u][0])
        
        for item_id_seen in items_to_filter:
            if 1 <= item_id_seen <= itemnum:  # Ensure item_id is valid
                scores[item_id_seen - 1] = np.inf # scores is 0-indexed

        # Calculate rank of the true_test_item
        # true_test_item is 1-indexed.
        # ranks_0based will give the 0-indexed rank for each item_id (0 to itemnum-1)
        ranks_0based = scores.argsort().argsort()
        rank = ranks_0based[true_test_item - 1]

        valid_user += 1

        for k in top_k_range:
            if rank < k:  # For top-k
                NDCG[k] += 1 / torch.log2(rank + 2)  # rank is 0-indexed
                HT[k] += 1

        if valid_user > 0 and valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user > 0 and usernum > 0 : # Print newline if dots were printed
        print("") 
    sys.stdout.flush()

    if valid_user == 0:
        return 0.0, 0.0

    return {k: v / valid_user for k, v in NDCG.items()}, {k: v / valid_user for k, v in HT.items()}