import lzma
import pickle
from typing import List, Iterable, Set, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, BertForTokenClassification
from pathlib import Path


class Solution:

    def __init__(self):

        LABELS_RAW = ['AGE', 'FAMILY', 'PENALTY', 'AWARD', 'IDEOLOGY',
                      'PERCENT', 'CITY', 'LANGUAGE', 'PERSON', 'COUNTRY', 'LAW',
                      'PRODUCT', 'CRIME', 'LOCATION', 'PROFESSION', 'DATE',
                      'MONEY', 'RELIGION', 'DISEASE', 'NATIONALITY',
                      'STATE_OR_PROV', 'DISTRICT', 'NUMBER', 'TIME', 'EVENT',
                      'ORDINAL', 'WORK_OF_ART', 'FACILITY', 'ORGANIZATION']

        LABELS_RAW.remove('STATE_OR_PROV')
        LABELS_RAW.append('STATE_OR_PROVINCE')  # why..

        self.LABELS = ['O'] + ['B_' + label for label in LABELS_RAW] + ['I_' + label
                                                                   for label in
                                                                   LABELS_RAW]

        TOKENIZER_PATH = Path(__file__).parent / 'tokenizer'
        MODEL_CONFIG_PATH = Path(__file__).parent / 'model_config'
        MODEL_WEIGHTS_PATH = Path(__file__).parent / 'model_state_dict'

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.model = self._load_model(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)

    def _get_tokens_from_text(self, s):
        # text -> tokens (numbers). Has 'offset_mappings'
        return self.tokenizer.encode_plus(s, return_offsets_mapping=True, return_tensors='pt')


    def _load_model(self, MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH):
        with open(MODEL_CONFIG_PATH, 'rb') as f:
            model_config_bin = f.read()
        with open(MODEL_WEIGHTS_PATH, 'rb') as f:
            model_weights_bin = f.read()
        model_config = pickle.loads(lzma.decompress(model_config_bin, format=lzma.FORMAT_RAW, filters=[{"id":lzma.FILTER_LZMA2,"dict_size":268435456, "preset":9, "mf":lzma.MF_HC3, "depth":0, "lc":3}]))
        model_weights = pickle.loads(lzma.decompress(model_weights_bin, format=lzma.FORMAT_RAW, filters=[{"id":lzma.FILTER_LZMA2,"dict_size":268435456, "preset":9, "mf":lzma.MF_HC3, "depth":0, "lc":3}]))
        return BertForTokenClassification.from_pretrained(config=model_config, state_dict=model_weights, pretrained_model_name_or_path=None)

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        for text in texts:
            tokens = self._get_tokens_from_text(text)
            tokens_ = tokens['offset_mapping'][0]
            model_output = self.model(tokens.input_ids)
            labels = model_output.logits.argmax(dim=2).tolist()[0]

            entities = set()
            current_type = ''
            current_from = 0
            current_to = 0
            for (a, b), label_int in zip(tokens_, labels):
                a, b = int(a), int(b)
                label = self.LABELS[label_int]
                if label == 'O':
                    if current_type:
                        entities.add((current_from, current_to, current_type))
                        current_type = ''
                    continue
                if label.startswith('B_'):
                    if current_type:
                        entities.add((current_from, current_to, current_type))
                    current_type = label[2:]
                    current_from = a
                    current_to = b
                    continue
                if label.startswith('I_'):
                    if label[2:] != current_type:
                        entities.add((current_from, current_to, current_type))
                        current_type = label[2:]
                        current_from = a
                        current_to = b
                    else:
                        current_to = b
                    continue

            yield entities
Solution()