from torch import nn
from transformers import BertModel, BertTokenizer
from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification
import transformers
import torch
from d2l import torch as d2l


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.pooler
        self.output = nn.Linear(768, 2)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))


if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    bert = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # print(bert)
    net = BERTClassifier(bert)



