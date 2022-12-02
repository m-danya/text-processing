# копировал из ноутбука, поэтому кодстайл такой странный

BANNED_NAMES = ['165459_text.txt', '176167_text.txt', '178485_text.txt', '192238_text.txt', '193267_text.txt', '193946_text.txt', '194112_text.txt', '2021.txt', '202294_text.txt', '2031.txt', '209438_text.txt', '209731_text.txt', '546860_text.txt']

LABELS_RAW = ['AGE', 'FAMILY', 'PENALTY', 'AWARD', 'IDEOLOGY', 'PERCENT', 'CITY', 'LANGUAGE', 'PERSON', 'COUNTRY', 'LAW', 'PRODUCT', 'CRIME', 'LOCATION', 'PROFESSION', 'DATE', 'MONEY', 'RELIGION', 'DISEASE', 'NATIONALITY', 'STATE_OR_PROV', 'DISTRICT', 'NUMBER', 'TIME', 'EVENT', 'ORDINAL', 'WORK_OF_ART', 'FACILITY', 'ORGANIZATION']

LABELS_RAW.remove('STATE_OR_PROV')
LABELS_RAW.append('STATE_OR_PROVINCE')  # поменяли почему-то

LABELS = ['O'] + ['B_' + label for label in LABELS_RAW] + ['I_' + label for label in LABELS_RAW]

len(LABELS)

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")


model = AutoModelForTokenClassification.from_pretrained("cointegrated/rubert-tiny2", num_labels=len(LABELS))
model.cuda()

from pathlib import Path


def ann_line_to_tuple(s):
    # ['T1', 'PERSON', '0', '15', 'Чулпан', 'Хаматова']
    _, t, a, b, text = s.rstrip().split(maxsplit=4)
    # different order to support sorting
    return int(a), int(b), t, text


def get_tokens_from_text(s):
    # text -> tokens (numbers). Has 'offset_mappings'
    return tokenizer.encode_plus(s, return_offsets_mapping=True)


def print_token(tokens_, tokens_idx, text):
    return
    print('TOKEN #', tokens_idx, ': ', tokens_[tokens_idx], '\t',
          text[tokens_[tokens_idx][0]:tokens_[tokens_idx][1]])


def print_label(labels_seq):
    return
    print('NEW LABEL', labels_seq[-1])


def load_seqs_from_dir(path_or_paths):
    if isinstance(path_or_paths, list):
        paths = path_or_paths
    else:
        paths = [path_or_paths]

    token_seqs, label_seqs = [], []

    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        for file_path in path.rglob("*.txt"):
            if file_path.name in BANNED_NAMES:
                continue
            with open(file_path) as f:
                text = f.read()
            with open(file_path.with_name(
                    file_path.name.replace('txt', 'ann'))) as f:
                spans_raw = [ann_line_to_tuple(line) for line in f if
                             line[0] == 'T' and ';' not in line]
                spans_raw.sort()
                spans = []
                for x in spans_raw:
                    if not any(x[0] <= y[0] and y[1] <= x[1] and x != y for y in
                               spans_raw):  # and not any(x[0] < y[0] and y[1] > x[1] and x != y for y in spans_raw):
                        spans.append(x)
            #             print(text)
            #             print('spans', list(spans[:30]))
            #             print('spans', list(spans))

            tokens = get_tokens_from_text(text)
            tokens_ = tokens['offset_mapping']
            if len(tokens_) > 2048:
                if 'train' in str(file_path):
                    # такой случай ОДИН в трейне
                    print('Warning: skipping 1 text because it is too long')
                    continue
                else:
                    raise Exception("Unexpected!")
            n_tokens = len(tokens_)
            labels_seq = []
            tokens_idx = 0
            # print(tokens_[:30])
            print_token(tokens_, tokens_idx, text)
            for span in spans:
                a, b, type_, span_text = span
                if a == b:
                    continue
                # print('span', a, b, type_, span_text)  # , text[a:b])

                #             print('span ', a, b, type_, span_text)
                # print('while conds', bool(tokens_[tokens_idx][0] < a),
                #       bool(tokens_[tokens_idx + 1][0] >= a))
                while tokens_[tokens_idx][0] < a and tokens_[tokens_idx + 1][0] <= b:
                    labels_seq.append('O')
                    print_label(labels_seq)
                    tokens_idx += 1
                    print_token(tokens_, tokens_idx, text)
                #             if tokens_[tokens_idx][0] != a:
                #                 print(text[tokens_[tokens_idx-1][0] : tokens_[tokens_idx-1][1]], text[tokens_[tokens_idx][0] : tokens_[tokens_idx][1]])
                #                 raise Exception("UNEXPECTED")
                labels_seq.append('B_' + type_)
                print_label(labels_seq)
                tokens_idx += 1
                print_token(tokens_, tokens_idx, text)
                while tokens_idx < n_tokens and tokens_[tokens_idx][1] <= b:
                    labels_seq.append('I_' + type_)
                    print_label(labels_seq)
                    tokens_idx += 1
                    print_token(tokens_, tokens_idx, text)
            while len(labels_seq) != n_tokens:
                # the rest of tokens (when spans have ended)
                labels_seq.append('O')
                print_label(labels_seq)
            # info = [(label, text[token[0]:token[1]]) for token, label in
            #         zip(tokens_, labels_seq) if label != 'O']
            #             print(info)
            token_seqs.append(tokens)
            label_seqs.append(labels_seq)
    #             break
    return token_seqs, label_seqs


train_tokens, train_labels = load_seqs_from_dir(Path(__file__).parent / 'nerel' / 'train')
dev_tokens, dev_labels = load_seqs_from_dir(Path(__file__).parent / 'nerel' / 'dev')
test_tokens, test_labels = load_seqs_from_dir(Path(__file__).parent / 'nerel' / 'test')

print("DATASETS HAVE BEEN PREPARED")

from typing import *
import torch

def get_label2idx(label_set: List[str]) -> Dict[str, int]:
    """
    Get mapping from labels to indices.
    """

    label2idx: Dict[str, int] = {}
    
    i = 0
    for label in label_set:
        label2idx[label] = i
        i += 1

    return label2idx

label2idx = get_label2idx(LABELS)

class TransformersDataset(torch.utils.data.Dataset):
    """
    Transformers Dataset for NER.
    """

    def __init__(
        self,
        token_seqs,
        label_seqs,
    ):
        self.token_seqs = token_seqs
        for t in self.token_seqs:
            if 'offset_mapping' in t:
                t.pop('offset_mapping')
        self.label_seqs = [self.process_labels(labels, label2idx) for labels in label_seqs]

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(
        self,
        idx: int,
    ):
#         return self.token_seqs[idx], self.label_seqs[idx]
        return {**self.token_seqs[idx], 'labels': self.label_seqs[idx]}
    
    @staticmethod
    def process_labels(
        labels: List[str],
        label2idx: Dict[str, int],
    ) -> List[int]:
        """
        Transform list of labels into list of labels' indices.
        """
        return torch.tensor([label2idx[label] for label in labels])

# train_dev_dataset = TransformersDataset(
#     train_dev_tokens,
#     train_dev_labels,
# )
    
train_dataset = TransformersDataset(
    train_tokens,
    train_labels,
)
dev_dataset = TransformersDataset(
    dev_tokens,
    dev_labels,
)
# test_dataset = TransformersDataset(
#     test_tokens,
#     test_labels,
# )

# print([len(x) for x in (train_dev_dataset, train_dataset, dev_dataset, test_dataset)])

from transformers import Trainer, TrainingArguments

import os
os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=50,              # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    learning_rate=8e-5,
    lr_scheduler_type='cosine',
    evaluation_strategy="epoch",
    save_strategy="no",
#     save_total_limit=2, # save only LAST and BEST models
#     load_best_model_at_end=False
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset             # evaluation dataset
)

print('gonna train')
trainer.train()
print('trained')

model.cpu()

import pickle
import lzma
print('gonna compress model files')
with open(Path(__file__).parent / 'model_state_dict', 'wb') as f:
    compressed = lzma.compress(pickle.dumps(model.state_dict()), format=lzma.FORMAT_RAW, filters=[{"id":lzma.FILTER_LZMA2,"dict_size":268435456, "preset":9, "mf":lzma.MF_HC3, "depth":0, "lc":3}])
    f.write(compressed)
with open(Path(__file__).parent / 'model_config', 'wb') as f:
    compressed = lzma.compress(pickle.dumps(model.config), format=lzma.FORMAT_RAW, filters=[{"id":lzma.FILTER_LZMA2,"dict_size":268435456, "preset":9, "mf":lzma.MF_HC3, "depth":0, "lc":3}])
    f.write(compressed)
tokenizer.save_pretrained(Path(__file__).parent / 'tokenizer')
print('DONE')
