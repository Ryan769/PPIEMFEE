import torch
import torch.nn as nn
import torch.nn.functional as F


# 层归一化
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

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
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.norm(x))


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

        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_size // self.n_heads]))
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.scale = torch.sqrt(torch.FloatTensor([self.hidden_size // self.n_heads])).to(device)

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


class BiLSTM_Attention_Layer(nn.Module):
    def __init__(self, hidden_size, n_layers, n_heads, dropout_rate):
        super(BiLSTM_Attention_Layer, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.bilstm = nn.LSTM(input_size=self.hidden_size, hidden_size=int(self.hidden_size // 2), num_layers=self.n_layers, 
                              bidirectional=True, batch_first=True)
        self.attention = MultiHeadSelfAttention(self.hidden_size, self.n_layers, self.dropout_rate)
        self.sublayer = SublayerConnection(self.hidden_size, self.dropout_rate)
    
    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.sublayer(out)
        out = self.attention(out, out, out)
        out = self.sublayer(out)
        return x


class CNNNet(nn.Module):
    # 多通道textcnn
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


input = torch.randn(6, 256, 390)
model = CNNNet(390, 3, '3,4,5', 0.1)

out = model(input)
print(out.shape)
        

        

