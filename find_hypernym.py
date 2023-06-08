import jieba
import re
import torch
from dataclasses import dataclass,field
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertPreTrainedModel

import clean_tex


@dataclass
class ModelArguments:
    use_lstm: bool = field(default=True, metadata={"help": "是否使用LSTM"})
    lstm_hidden_size: int = field(default=500, metadata={"help": "LSTM隐藏层输出的维度"})
    lstm_layers: int = field(default=1, metadata={"help": "堆叠LSTM的层数"})
    lstm_dropout: float = field(default=0.5, metadata={"help": "LSTM的dropout"})
    hidden_dropout: float = field(default=0.5, metadata={"help": "预训练模型输出向量表示的dropout"})
    ner_num_labels: int = field(default=22, metadata={"help": "需要预测的标签数量"})

@dataclass
class Example:
    text: List[str] # ner的文本
    label: List[str] = None # ner的标签

    def __post_init__(self):
        if self.label:
            assert len(self.text) == len(self.label)

def get_label_from_list():
    return ["<pad>", "B-CON", "I-CON", "B-OPR", "I-OPR",
            "B-ATT", "I-ATT", "B-ATV", "I-ATV", "B-REL", "I-REL", "B-ASS", "I-ASS",
            "S-CON","S-OPR","S-ATT","S-ATV","S-REL","S-ASS",
            'O', "<start>", "<eos>"]

def load_tag():
    tags=get_label_from_list()
    id2tag = {i: label for i, label in enumerate(tags)}
    tag2id = {label: i for i, label in enumerate(tags)}
    return id2tag,tag2id


class NERDataset(Dataset):
    def __init__(self, examples: List[Example], max_length=128,
                 tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')):
        self.max_length = 512 if max_length > 512 else max_length
        self.texts = [torch.LongTensor(tokenizer.encode(example.text[: self.max_length - 2])) for example in examples]
        self.labels = []
        for example in examples:
            label = example.label
            """
            1. 将字符的label转换为对于的id；
            2. 控制label的最长长度；
            3. 添加开始位置和结束位置对应的标签，这里<start>对应输入中的[CLS],<eos>对于[SEP]；
            4. 转换为Tensor；
            """
            label = [tag2id["<start>"]] + [tag2id[l] for l in label][: self.max_length - 2] + [tag2id["<eos>"]]
            self.labels.append(torch.LongTensor(label))
        assert len(self.texts) == len(self.labels)
        for text, label in zip(self.texts, self.labels):
            assert len(text) == len(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "input_ids": self.texts[item],
            "labels": self.labels[item]
        }

class BertForNER(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config) # 初始化父类(必要的步骤)
        if "model_args" in model_kargs:
            model_args = model_kargs["model_args"]
            """
            必须将额外的参数更新至self.config中，这样在调用save_model保存模型时才会将这些参数保存；
            这种在使用from_pretrained方法加载模型时才不会出错；
            """
            self.config.__dict__.update(model_args.__dict__)
        self.num_labels = self.config.ner_num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.config.hidden_dropout)
        self.lstm = nn.LSTM(self.config.hidden_size, # 输入的维度
                            self.config.lstm_hidden_size, # 输出维度
                            num_layers=self.config.lstm_layers, # 堆叠lstm的层数
                            dropout=self.config.lstm_dropout,
                            bidirectional=True, # 是否双向
                            batch_first=True)
        if self.config.use_lstm:
            self.classifier = nn.Linear(self.config.lstm_hidden_size * 2, self.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.dropout(outputs[0])
        if self.config.use_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 如果attention_mask不为空，则只计算attention_mask中为1部分的Loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

def trans_text(text):
    pre_data = [Example(list(text), ['O'] * len(text))]
    pre_data = NERDataset(pre_data)
    pre_data = DataLoader(pre_data, batch_size=1)
    pre_data = next(iter(pre_data))
    return pre_data

def get_entities(words,labels):
    entyties = []
    entity = []
    flag = 0
    for i in range(len(labels)):
        if labels[i]!='O':
            flag = 1
            entity.append(words[i])
        elif flag==1:
            if len(entity)>0:
                entyties.append(''.join(entity))
                entity = []
    return entyties


level1 = ''
level2 = ''
if __name__ == '__main__':
    id2tag, tag2id = load_tag()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model_args = ModelArguments(use_lstm=True)
    checkpoint = './ner_model_best'
    model = BertForNER.from_pretrained(checkpoint, model_args=model_args)
    with open('./data/cleaned_tex.tex') as file:
        for line in file:
            if re.match(r'{第\d+章([\u2E80-\u9FFF]+)}',line):
                level1 = re.match(r'{第\d+章([\u2E80-\u9FFF]+)}',line).group(1)
                #print(f'level1:{level1} '+'*'*15)
            elif re.match(r'{.*\d+\..+}',line):
                level2 = re.match(r'{.*\d+\.(.+)}',line).group(1)
                #print(f'level2:{level2} ' + '*' * 8)
            else:
                line = line.strip()
                line = line.strip('。')
                texts = line.split('。')
                entities = []
                for text in texts:
                    pre_data = trans_text(text)
                    _, logits = model(**pre_data)
                    res = torch.argmax(logits[0], 1)
                    res = [id2tag[item] for item in res.tolist()]
                    res = res[1:-1]
                    ans = get_entities(words=list(text),labels=res)
                    entities += ans
                entities = set(entities)
                #去除无效实体（非中文）
                entities = {entity for entity in entities if clean_tex.calculate_chinese_ratio(entity)>0.6}
                print(entities)
                