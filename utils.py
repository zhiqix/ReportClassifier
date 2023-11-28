from torch.utils import data
from config import *
import torch
from transformers import AutoTokenizer
from sklearn.metrics import classification_report


class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'dev':
            sample_path = DEV_SAMPLE_PATH

        self.lines = open(sample_path, encoding='gbk').readlines()
        self.tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        str_list = self.lines[index].split(',')
        new_str = ""
        if len(str_list) == 2:
            new_str = str_list[0]
        else:
            # Replace the English commas in the original text with Chinese commas.
            for i in range(len(str_list) - 2):
                new_str += str_list[i] + "，"
            new_str += str_list[-2]

        text = new_str
        target = int(str_list[-1])

        tokened = self.tokenizer(text, padding=True, truncation=True)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        # Pad the input_ids to ensure that the length of the input_ids reaches TEXT_LEN.
        if len(input_ids) < TEXT_LEN:
            pad_len = TEXT_LEN - len(input_ids)
            input_ids += [BGE_PAD_ID] * pad_len
            mask += [0] * pad_len

        return torch.tensor(input_ids[:TEXT_LEN]), torch.tensor(mask[:TEXT_LEN]), torch.tensor(target)


def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        true,
        pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )
