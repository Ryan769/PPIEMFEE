import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class charEmbedding(nn.Module):
    """
    Input: (max_length, max_character_length) max_length 是指句子的最大长度，也就是char的batch size
    Output: (max_length, filter_num)
    """
    def __init__(self, args):
        super(charEmbedding, self).__init__()
        self.args = args
        self.emb = nn.Embedding(self.args.char_vocab_size, self.args.char_embedding_size)
        self.conv = nn.Conv1d(in_channels=self.args.char_embedding_size, out_channels=self.args.filter_number,
                              kernel_size=self.args.kernel_size)
        self.pool = torch.nn.MaxPool1d(self.args.max_character_len - self.args.kernel_size + 1, stride=1)
        self.drop = torch.nn.Dropout(self.args.dropout_rate)

    def forward(self, x):
        """
            x: one char sequence. shape: (max_len, max_character_len)
        """
        # 如果输入是一句话
        inp = self.drop(self.emb(x))  # (max_len, max_character_len) -> (max_len, max_character_len, hidden)
        inp = inp.permute(0, 2, 1)  # (max_len, max_character_len, hidden) -> (max_len,  hidden, max_character_len)
        out = self.conv(inp)  # out: (max_len, filter_num, max_character_len - kernel_size + 2)
        return self.pool(out).squeeze()  # out: (max_len, filter_num)


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(self.norm(x))


class CNNNet(nn.Module):

    def __init__(self, hidden_size, filter_num, filter_sizes, dropout_rate):
        super(CNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate

        filter_size_list = [int(fsz) for fsz in self.filter_sizes.split(',')]  # [3,4,5]
        self.convs = nn.ModuleList([nn.Conv2d(1, self.filter_num, (fsz, hidden_size)) for fsz in filter_size_list])

        # dropout是在训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear = nn.Linear(len(filter_size_list) * filter_num, hidden_size)

    def forward(self, x):

        # 经过view函数x的维度变为(batch_size, input_chanel=1, seq_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.hidden_size)

        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算得到的多维度张量展平
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来
        x = torch.cat(x, 1)

        # dropout层
        x = self.dropout(x)

        # 全连接层
        logits = self.linear(x)

        return logits


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.drop_prob = drop_prob

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert self.hidden_size % self.n_heads == 0

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_size // self.n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        # Q,K,V计算与变形：
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hidden_size // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hidden_size // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hidden_size // self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.dropout(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘
        sequence_output = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了
        sequence_output = sequence_output.permute(0, 2, 1, 3).contiguous()
        sequence_output = sequence_output.view(bsz, -1, self.n_heads * (self.hidden_size // self.n_heads))
        sequence_output = self.fc(sequence_output)
        return sequence_output


class BioBERTLayer(nn.Module):
    def __init__(self, config, args):
        super(BioBERTLayer, self).__init__()
        self.args = args
        self.biobert = BertModel.from_pretrained(self.args.model_name_or_path)
        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.sublayer = SublayerConnection(config.hidden_size * 3, args.dropout_rate)
        self.linear = nn.Linear(config.hidden_size * 3, config.hidden_size * 3)

        self.dropout = nn.Dropout(args.dropout_rate)

    def masked_avgpool(self, sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+2, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 2, 2, 2, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 2, j-i+2]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 2]

        # [b, 2, j-i+2] * [b, j-i+2, dim] = [b, 2, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask):
        outputs = self.biobert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = self.masked_avgpool(sequence_output, attention_mask)   # 对bert和bilstm拼接的向量做平均池化
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer -> global_feature
        output_feature = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        # output_feature = pooled_output

        output_feature = self.dropout(output_feature)
        output_feature = self.sublayer(output_feature)
        output_feature = self.linear(output_feature)

        return output_feature


class BiLSTM_ResNet_Attention_Layer(nn.Module):
    def __init__(self, args):
        super(BiLSTM_ResNet_Attention_Layer, self).__init__()
        self.hidden_size = args.word_embedding_size + args.filter_number + args.pos_embedding_size

        # 各种特征嵌入
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.word_embedding_size)
        self.char_encode = charEmbedding(args)
        self.pos_embedding = nn.Embedding(args.pos_vocab_size, args.pos_embedding_size, padding_idx=0)

        # bilstm
        self.bilstm = nn.LSTM(input_size=self.hidden_size, hidden_size=int(self.hidden_size // 2), num_layers=args.bilstm_layers,
                              bidirectional=True, batch_first=True)

        # attention
        self.attention = MultiHeadSelfAttention(self.hidden_size, args.n_heads, args.dropout_rate)
        self.cnnnet = CNNNet(self.hidden_size, args.filter_num, args.filter_sizes, args.dropout_rate)
        self.entity_fc_layer = FCLayer(self.hidden_size, self.hidden_size, args.dropout_rate)

        # self.sublayer = SublayerConnection(self.hidden_size * 3, args.dropout_rate)
        # self.linear = nn.Linear(self.hidden_size * 3, self.hidden_size * 3)

        # no_entity_word-level_feature
        self.sublayer = SublayerConnection(self.hidden_size, args.dropout_rate)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(args.dropout_rate)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.pos_embedding.weight.data)
        nn.init.kaiming_uniform_(self.char_encode.weigth.data)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+2, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 2, 2, 2, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 2, j-i+2]

        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 2]

        # [b, 2, j-i+2] * [b, j-i+2, dim] = [b, 2, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, char_ids, pos_ids, e1_mask, e2_mask):
        word_emb = self.dropout(self.word_embedding(input_ids))
        char_embedding = []
        for i in range(char_ids.shape[0]):
            one_word_char_emb = self.char_encode(char_ids[i])
            char_embedding.append(one_word_char_emb)
        char_emb = torch.stack(char_embedding)
        pos_emb = self.pos_embedding(pos_ids)
        word_emb = torch.cat((word_emb, char_emb, pos_emb), -1)
        sequence_output, _ = self.bilstm(word_emb)
        sequence_output = self.attention(sequence_output, sequence_output, sequence_output)

        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cnnnet(sequence_output)   # 对bilstm的向量做卷积
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer -> local_feature
        # output_feature = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        output_feature = pooled_output

        output_feature = self.dropout(output_feature)
        output_feature = self.sublayer(output_feature)
        output_feature = self.linear(output_feature)

        return output_feature


class Global_Local_FusedLayer(nn.Module):
    def __init__(self, config, args):
        super(Global_Local_FusedLayer, self).__init__()
        self.num_labels = config.num_labels

        # self.hidden_size = config.hidden_size + args.word_embedding_size + args.filter_number + args.pos_embedding_size
        # self.label_classifier = FCLayer(self.hidden_size * 3, self.num_labels, args.dropout_rate, use_activation=False)
        # self.label_classifier1 = FCLayer(config.hidden_size * 3, self.num_labels, args.dropout_rate, use_activation=False)
        # self.label_classifier2 = FCLayer((args.word_embedding_size + args.filter_number + args.pos_embedding_size) * 3, self.num_labels, args.dropout_rate, use_activation=False)

        # # no entity_contextul_feature
        # self.hidden_size = config.hidden_size + (args.word_embedding_size + args.filter_number + args.pos_embedding_size) * 3
        # self.label_classifier = FCLayer(self.hidden_size, self.num_labels, args.dropout_rate, use_activation=False)
        # self.label_classifier1 = FCLayer(config.hidden_size, self.num_labels, args.dropout_rate, use_activation=False)
        # self.label_classifier2 = FCLayer((args.word_embedding_size + args.filter_number + args.pos_embedding_size) * 3, self.num_labels, args.dropout_rate, use_activation=False)

        # no entity_word-level_feature
        self.hidden_size = config.hidden_size * 3 + args.word_embedding_size + args.filter_number + args.pos_embedding_size
        self.label_classifier = FCLayer(self.hidden_size, self.num_labels, args.dropout_rate, use_activation=False)
        self.label_classifier1 = FCLayer(config.hidden_size * 3, self.num_labels, args.dropout_rate, use_activation=False)
        self.label_classifier2 = FCLayer(args.word_embedding_size + args.filter_number + args.pos_embedding_size, self.num_labels, args.dropout_rate, use_activation=False)

        self.biobert = BioBERTLayer(config, args)
        self.bilstm_res_att_layer = BiLSTM_ResNet_Attention_Layer(args)

    def forward(self, input_ids, attention_mask, token_type_ids, char_ids, pos_ids, labels, e1_mask, e2_mask):
        # Concat -> fc_layer
        global_feature = self.biobert(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask)
        local_feature = self.bilstm_res_att_layer(input_ids, char_ids, pos_ids, e1_mask, e2_mask)
        concat_h = torch.cat([global_feature, local_feature], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (logits,)

        # 消融
        # 去除local_feature
        # global_feature = self.biobert(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask)
        # logits = self.label_classifier1(global_feature)
        # outputs = (logits,)

        # 去除global_feature
        # local_feature = self.bilstm_res_att_layer(input_ids, char_ids, pos_ids, e1_mask, e2_mask)
        # logits = self.label_classifier2(local_feature)
        # outputs = (logits,)

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # 二进制交叉熵函数
                # loss = F.binary_cross_entropy_with_logits(logits.view(-2, self.num_labels), labels.view(-2))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits
