import torch
import torch.nn as nn

from torchcrf import CRF
from transformers import BertModel, BertConfig

import torch.nn.functional as F
class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # Calculate the attention weights
        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)
        energy = self.energy_layer(torch.tanh(query + key))
        attention_weights = F.softmax(energy, dim=1)

        # Get the weighted representations
        weighted_representations = attention_weights * hidden_states

        return weighted_representations, attention_weights


class BertNer(nn.Module):

    # 下面是消融了BILSTM层的INIT函数

    # def __init__(self, args):
    #     super(BertNer, self).__init__()
    #     self.bert = BertModel.from_pretrained(args.bert_dir)
    #     self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    #     hidden_size = self.bert_config.hidden_size
    #     self.max_seq_len = args.max_seq_len
    #     self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)  # 添加多头自注意力
    #     self.linear = nn.Linear(hidden_size, args.num_labels)
    #     self.crf = CRF(args.num_labels, batch_first=True)
    def __init__(self, args):
        super(BertNer, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        hidden_size = self.bert_config.hidden_size
        # 隐藏层维度
        self.lstm_hiden = 384
        self.max_seq_len = args.max_seq_len
        self.bigru = nn.GRU(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True, dropout=0.1)
        # self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
        #            dropout=0.1)
        # self.additive_attention = AdditiveAttention(self.lstm_hiden * 2)
        # 多头自注意力
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=self.lstm_hiden * 2, num_heads=8)
        self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
        self.crf = CRF(args.num_labels, batch_first=True)
    #
    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_output[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)
        seq_out, _ = self.bigru(seq_out)
        seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
        seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
        seq_out = self.linear(seq_out)
        logits = self.crf.decode(seq_out, mask=attention_mask.bool())
        loss = None
        if labels is not None:
            loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
        model_output = ModelOutput(logits, labels, loss)
        return model_output

    # 下面是带self-attention的反向传播函数
    # def forward(self, input_ids, attention_mask, labels=None):
    #     bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #     seq_out = bert_output[0]  # [batchsize, max_len, 768]
    #     batch_size = seq_out.size(0)
    #     seq_out, _ = self.bigru(seq_out)
    #
    #     # Reformat seq_out for MultiheadAttention: [len, batch, dim]
    #     seq_out = seq_out.permute(1, 0, 2)
    #     attn_output, _ = self.multihead_attention(seq_out, seq_out, seq_out, key_padding_mask=~attention_mask.bool())
    #     seq_out = attn_output.permute(1, 0, 2)  # Reformat back
    #
    #     seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    #     seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
    #     seq_out = self.linear(seq_out)
    #     logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    #     loss = None
    #     if labels is not None:
    #         loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    #     model_output = ModelOutput(logits, labels, loss)
    #     return model_output

    # 此方法为加性注意力的实现方法
    # def forward(self, input_ids, attention_mask, labels=None):
    #     bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #     seq_out = bert_output[0]  # [batchsize, max_len, 768]
    #     batch_size = seq_out.size(0)
    #     seq_out, _ = self.bilstm(seq_out)
    #     # Apply additive attention
    #     seq_out, attention_weights = self.additive_attention(seq_out)
    #
    #     seq_out = self.linear(seq_out)
    #     logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    #     loss = None
    #     if labels is not None:
    #         loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    #     model_output = ModelOutput(logits, labels, loss)
    #     return model_output

    # 消融了bilstm的反向传播函数
    # def forward(self, input_ids, attention_mask, labels=None):
    #     bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #     seq_out = bert_output[0]  # [batchsize, max_len, 768]
    #
    #     # 对多头自注意力进行调整
    #     attn_output, _ = self.multihead_attention(seq_out.transpose(0, 1), seq_out.transpose(0, 1),
    #                                               seq_out.transpose(0, 1), key_padding_mask=~attention_mask.bool())
    #     seq_out = attn_output.transpose(0, 1)
    #
    #     seq_out = self.linear(seq_out)
    #     logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    #     loss = None
    #     if labels is not None:
    #         loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    #     model_output = ModelOutput(logits, labels, loss)
    #     return model_output

