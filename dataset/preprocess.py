import argparse
import time
import csv
import operator
import datetime
import os
from tqdm import tqdm
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/nowplaying/retailrocket/tmall/yoochoose')
opt = parser.parse_args()
print(opt)


def is_same_day(t1, t2):
    if t1 >= 10000000000:
        t1 /= 1000
    if t2 >= 10000000000:
        t2 /= 1000
    t1 = datetime.datetime.fromtimestamp(t1)
    t2 = datetime.datetime.fromtimestamp(t2)
    return t1.date() == t2.date()


def convert_retailrocket():
    user2seq = {}
    with open('raw/events.csv', 'r') as file:
        file.readline()
        for line in file:
            timestamp, visitorid, event, itemid, transactionid = line.strip().split(',')
            if event != 'view':
                continue
            if visitorid not in user2seq:
                user2seq[visitorid] = []
            user2seq[visitorid].append((int(timestamp), itemid))
    for user in tqdm(user2seq, desc='Sorting user seqs:'):
        user2seq[user].sort(key=lambda t: t[0])
    tot_sess_id = 0
    with open('raw/retailrocket.csv', 'w') as file:
        file.write('session_id,timestamp,item_id\n')
        for user in tqdm(user2seq, desc='Converting to sessions'):
            tot_sess_id += 1
            last_ts = user2seq[user][0][0]
            for ts, itemid in user2seq[user]:
                if not is_same_day(last_ts, ts):
                    tot_sess_id += 1
                last_ts = ts
                file.write(f'{tot_sess_id},{ts},{itemid}\n')


dataset = None
if opt.dataset == 'diginetica':
    dataset = 'raw/train-item-views.csv'
elif opt.dataset == 'nowplaying':
    dataset = 'raw/nowplaying.csv'
elif opt.dataset == 'retailrocket':
    convert_retailrocket()
    dataset = 'raw/retailrocket.csv'
elif opt.dataset == 'tmall':
    dataset = 'raw/dataset15.csv'
elif opt.dataset =='yoochoose':
    dataset = 'raw/yoochoose.csv'


print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset in ['retailrocket', 'yoochoose']:
        reader = csv.DictReader(f, delimiter=',')
    elif opt.dataset in ['diginetica']:
        reader = csv.DictReader(f, delimiter=';')
    elif opt.dataset in ['nowplaying', 'tmall']:
        reader = csv.DictReader(f, delimiter='\t')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        if opt.dataset in ['tmall', 'nowplaying']:
            sessid = data['SessionId']
        elif opt.dataset in ['diginetica', 'yoochoose', 'retailrocket']:
            sessid = data['session_id']
        else:
            raise ValueError()
        if opt.dataset == 'tmall' and int(sessid) > 120000:
            continue
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            elif opt.dataset == 'diginetica':
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            elif opt.dataset in ['retailrocket']:
                date = float(curdate) / 1000.
            elif opt.dataset in ['tmall', 'nowplaying']:
                date = float(curdate)
            else:
                raise ValueError()
            sess_date[curid] = date
        curid = sessid
        if opt.dataset in ['yoochoose', 'retailrocket']:
            item = data['item_id']
        elif opt.dataset == 'diginetica':
            item = data['item_id'], int(data['timeframe'])
        elif opt.dataset in ['tmall', 'nowplaying']:
            item = data['ItemId']
        else:
            raise ValueError()
        curdate = ''
        if opt.dataset in ['yoochoose', 'retailrocket']:
            curdate = data['timestamp']
        elif opt.dataset == 'diginetica':
            curdate = data['eventdate']
        elif opt.dataset in ['tmall', 'nowplaying']:
            curdate = data['Time']
        else:
            raise ValueError()
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    elif opt.dataset == 'diginetica':
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    elif opt.dataset in ['retailrocket']:
        date = float(curdate) / 1000.
    elif opt.dataset in ['tmall', 'nowplaying']:
        date = float(curdate)
    else:
        raise ValueError()
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out sessions by length
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())

print('sorting %ss' % datetime.datetime.now())
dates.sort(key=lambda t: t[1])
print('finish sorting %ss' % datetime.datetime.now())


if opt.dataset == 'yoochoose':
    split4 = int(len(dates) / 4.)
    dates = dates[-split4:]
    print('1/4 remained.')

tot = len(dates)
train_split = int(tot * 0.8)
valid_split = int(tot * 0.9)
train_slice = slice(0, train_split)
valid_slice = slice(train_split, valid_split)
test_slice = slice(valid_split, tot)

train_dates, valid_dates, test_dates = dates[train_slice], dates[valid_slice], dates[test_slice]
print(len(train_dates), len(valid_dates), len(test_dates))

print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra(tra_sess):
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(f'# Items {item_ctr}')     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes(tes_sess):
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra(train_dates)
val_ids, val_dates, val_seqs = obtian_tes(valid_dates)
tes_ids, tes_dates, tes_seqs = obtian_tes(test_dates)


print(f'# Sessions {len(tra_seqs) + len(val_seqs) + len(tes_seqs)}')


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i][-50:]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
va_seqs, va_dates, va_labs, va_ids = process_seqs(val_seqs, val_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)


os.mkdir(f'{opt.dataset}')
n_inters = 0
with open(f'{opt.dataset}/{opt.dataset}.train.inter', 'w') as file:
    file.write('\t'.join(['session_id:token', 'item_id_list:token_seq', 'item_id:token']) + '\n')
    for i in range(len(tr_ids)):
        file.write(f'{n_inters + i + 1}\t{" ".join(map(str, tr_seqs[i]))}\t{tr_labs[i]}\n')
n_inters += len(tr_ids)

with open(f'{opt.dataset}/{opt.dataset}.valid.inter', 'w') as file:
    file.write('\t'.join(['session_id:token', 'item_id_list:token_seq', 'item_id:token']) + '\n')
    for i in range(len(va_ids)):
        file.write(f'{n_inters + i + 1}\t{" ".join(map(str, va_seqs[i]))}\t{va_labs[i]}\n')
n_inters += len(va_ids)

with open(f'{opt.dataset}/{opt.dataset}.test.inter', 'w') as file:
    file.write('\t'.join(['session_id:token', 'item_id_list:token_seq', 'item_id:token']) + '\n')
    for i in range(len(te_ids)):
        file.write(f'{n_inters + i + 1}\t{" ".join(map(str, te_seqs[i]))}\t{te_labs[i]}\n')
n_inters += len(te_ids)


print(f'# Interactions {n_inters}')

ave_l_sess = sum([len(_) for _ in tr_seqs + va_seqs + te_seqs]) / n_inters
print(f'Avg. Length {ave_l_sess}')
