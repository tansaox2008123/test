import os
import re
import datetime
import time
import numpy as np
import random
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.nn.functional as F
import math
from multiprocessing import Pool
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
import os  # 打开读取文件
# import cv2  opencv读取图像
from matplotlib import pyplot as plt  # 测试图像是否读入进行绘制图像
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ast


def default_loader(path):
    return np.load(path)


def custom_transform(tensor):
    tensor = np.array(tensor).flatten()
    print(1)
    return tensor


class MyDataset(Dataset):

    def __init__(self, txt, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        rnas = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:
            words = line.split()
            b = words[1:]
            a = ''.join(b)
            match = re.search(r'\[.*\]', a)
            if match:
                # 获取匹配到的部分
                substring = match.group(0)
                # 将字符串表示的列表转换为实际的列表
                values_list = ast.literal_eval(substring)
                # 将列表转换为张量
                tensor_from_values = torch.tensor(values_list)
                tensor_from_values = tensor_from_values
            else:
                tensor_from_values = None
            rnas.append((words[0], tensor_from_values))
            self.rnas = rnas
            # 将（20，640）转化为12800
            # 使用 transforms.Compose 将自定义转换函数包括进来
            # 应用转换
            self.transform = transform
            # self.transform = transforms.Compose([transforms.Lambda(custom_transform)])
            self.loader = loader

    def __getitem__(self, index):
        fn, label = self.rnas[index]
        # fn是rna的path
        rna = self.loader(fn)
        rna = np.sum(rna, axis=0)
        rna = rna
        print("Loaded Data Shape:", rna.shape)
        # print(self.num)
        # self.num = self.num + 1
        # 按照路径读取rna
        if self.transform is not None:
            print("1225")
            rna = self.transform(rna)
        return rna, label

    def __len__(self):
        return len(self.rnas)


def MyDataLoader(directory_name, tag_path, length, test_rate, batch_size):
    # create_txt(directory_name, tag_path, length, test_rate)
    train_data = MyDataset(txt='/home2/public/data/RNA_aptamer/data/sample_train2.txt')
    test_data = MyDataset(txt='/home2/public/data/RNA_aptamer/data/sample_test2.txt')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    return train_loader, test_loader


# Attention 计算
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q:batch_size, n_heads, len_q, d_k
        # K:batch_size, n_heads, len_k, d_k
        # V:batch_size, n_heads, len_v, d_v
        # attn_mask:batch_size, n_heads, seq_len, seq_len

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # scores:batch_size, n_heads, len_q, len_k
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


# tgt_size  projection线性层的输出维度，这里输出的是4维向量对应四个碱基
tgt_size = 20
src_size = 20
# 占位使用，后期修改
# src_vocab_size = 20
#
# tgt_vocab_size = 20
#
# src_len = 20  # length of source
# tgt_len = 20  # length of target

# 模型参数
d_model = 64  # Embedding Size
d_ff = 128  # FeedForward dimension
d_k = d_v = 32  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention


def get_attention_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    # 最终得到的应该是一个最后n列为1的矩阵，即K的最后n个token为PAD。
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


# Sequence Mask 屏蔽未来词，构成上三角矩阵
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    # np.triu()返回一个上三角矩阵，自对角线k以下元素全部置为0，k=0为主对角线
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    # 如果没转成byte，这里默认是Double(float64)，占据的内存空间大，浪费，用byte就够了
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attention_mask):
        # attention_mask:batch_size,len_q,len_k
        # Q:batch_size,len_q,d_model
        # Q:batch_size,len_k,d_model
        # Q:batch_size,len_k,d_model
        residual, batch_size = Q, Q.size(0)

        # q_s:batch_size,n_heads,len_q,d_k
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # k_s:batch_size,n_heads,len_k,d_k
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # v_s:batch_size,n_heads,len_k,d_v
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attention_mask:batch_size,len_q,len_k ----> batch_size,n_heads,len_q,len_k
        attention_mask = attention_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attention = ScaledDotProductAttention(d_k)(q_s, k_s, v_s, attention_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)

        return self.layer_norm(output + residual), attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 生成一个形状为[max_len,d_model]的全为0的tensor
        pe = torch.zeros(max_len, d_model)
        # position:[max_len,1]，即[5000,1]，这里插入一个维度是为了后面能够进行广播机制然后和div_term直接相乘
        # 注意，要理解一下这里position的维度。每个pos都需要512个编码。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 共有项，利用指数函数e和对数函数log取下来，方便计算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 这里position * div_term有广播机制，因为div_term的形状为[d_model/2],即[256],符合广播条件，广播后两个tensor经过复制，形状都会变成[5000,256]，*表示两个tensor对应位置处的两个元素相乘
        # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置赋值给pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # 同理，这里是奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        # 上面代码获取之后得到的pe:[max_len*d_model]

        # 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 定一个缓冲区，其实简单理解为这个参数不更新就可以，但是参数仍然作为模型的参数保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # 这里的self.pe是从缓冲区里拿的
        # 切片操作，把pe第一维的前seq_len个tensor和x相加，其他维度不变
        # 这里其实也有广播机制，pe:[max_len,1,d_model]，第二维大小为1，会自动扩张到batch_size大小。
        # 实现词嵌入和位置编码的线性相加
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PoswiseFeedForwardNet2(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        # Conv1d的输入为[batch, channel, length]，作用于第二个维度channel，所以这里要转置
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)


# enc_inputs:batch_size,src_len
# dec_inputs:batch_size,tgt_len
# enc_outputs:batch_size,src_len,d_model
# enc_self_attention:batch_size,n_heads,src_len,src_len
# dec_outputs:batch_size,tgt_len,d_model
# dec_self_attention:batch_size,n_heads,tgt_len,src_len
# dec_enc_attention:batch_size,n_heads,tgt_len,src_len
# dec_logits:batch_size,tgt_len,tgt_vocab_size
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_size, bias=False)
        # d_model,tgt_size是自定义全局参数

    def foward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attention = self.encoder(enc_inputs)
        dec_outputs, dec_self_attention, dec_enc_attention = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs映射到词表大小
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attention, dec_self_attention, dec_enc_attention


class TransformerDecoderOnly(nn.Module):
    def __init__(self, input_size11, hidden_size22):
        super(TransformerDecoderOnly, self).__init__()

        self.hidden1 = nn.Linear(input_size11, hidden_size22)
        self.relu1 = nn.ReLU()

        self.decoder2 = Decoder2()
        # 这一行代码定义了一个线性层，称为projection，它接收的输入特征维度为 d_model，输出特征维度为 tgt_size
        self.projection = nn.Linear(d_model, tgt_size, bias=False)

    def forward(self, dec_inputs):
        x = dec_inputs
        x = x.to(self.hidden1.weight.dtype)
        x = self.hidden1(x)
        x = self.relu1(x)

        x = torch.tensor(x).to(torch.long)

        dec_outputs, dec_self_attns = self.decoder2(x)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 词嵌入矩阵
        self.src_emb = nn.Embedding(src_size, d_model)
        # 位置编码
        self.pos_emb = PositionalEncoding(d_model)
        # [EncoderLayer() for _ in range(n_layers)]：这是一个列表推导式，
        # 创建了包含 n_layers 个 EncoderLayer 实例的列表。通过循环 n_layers 次，为每个循环创建一个新的 EncoderLayer 实例，并将其添加到列表中。
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # 通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)

        # 位置编码和词嵌入相加，具体实现在PositionalEncoding里，enc_outputs:[batch_size,src_len,d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        enc_self_attention_mask = get_attention_pad_mask(enc_inputs, enc_inputs)
        enc_self_attentions = []
        for layer in self.layers:
            enc_outputs, enc_self_attention = layer(enc_outputs, enc_self_attention_mask)
            enc_self_attentions.append(enc_self_attention)

        return enc_outputs, enc_self_attentions


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attention = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attention_mask):
        # 三个enc_inputs分别为Q，K，V
        enc_outputs, attention = self.enc_self_attention(enc_inputs, enc_inputs, enc_inputs, enc_self_attention_mask)
        # enc_outputs:batch_size,len_q,d_model
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attention


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    # dec_inputs:batch_size,target_len
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # 得到的dec_outputs维度为  dec_outputs:batch_size,tgt_len,d_model
        dec_outputs = self.tgt_emb(dec_inputs)
        # 得到的dec_outputs维度为  dec_outputs:batch_size,tgt_len,d_model
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        # 获取自注意力的pad_mask，1表示被mask
        dec_self_attention_pad_mask = get_attention_pad_mask(dec_inputs, dec_inputs)

        # 获取上三角矩阵，即让注意力机制看不到未来的单词，1表示被mask
        dec_self_attention_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 两个mask矩阵相加，大于0的为1，不大于0的为0，即屏蔽了pad的信息，也屏蔽了未来时刻的信息，为1的在之后就会被fill到无限小
        dec_self_attention_mask = torch.gt((dec_self_attention_subsequent_mask + dec_self_attention_pad_mask), 0)

        dec_enc_attention_mask = get_attention_pad_mask(dec_inputs, enc_inputs)

        dec_self_attentions, dec_enc_attentions = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attention, dec_enc_attention = layer(dec_outputs, enc_outputs,
                                                                       dec_self_attention_mask, dec_enc_attention_mask)
            dec_self_attentions.append(dec_self_attention)
            dec_enc_attentions.append(dec_enc_attention)
        return dec_outputs, dec_self_attentions, dec_enc_attentions


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer2() for _ in range(n_layers)])

    # dec_inputs:batch_size,target_len
    def forward(self, dec_inputs):
        # 得到的dec_outputs维度为  dec_outputs:batch_size,tgt_len,d_model
        dec_outputs = self.tgt_emb(dec_inputs)
        # 得到的dec_outputs维度为  dec_outputs:batch_size,tgt_len,d_model
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        # 获取自注意力的pad_mask，1表示被mask
        # dec_self_attention_pad_mask = get_attention_pad_mask(dec_inputs, dec_inputs).to(device)

        # 获取上三角矩阵，即让注意力机制看不到未来的单词，1表示被mask
        dec_self_attention_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)

        # 两个mask矩阵相加，大于0的为1，不大于0的为0，即屏蔽了pad的信息，也屏蔽了未来时刻的信息，为1的在之后就会被fill到无限小
        # dec_self_attention_mask = torch.gt((dec_self_attention_subsequent_mask + dec_self_attention_pad_mask), 0)

        dec_self_attention_mask = torch.gt(dec_self_attention_subsequent_mask, 0)

        # dec_enc_attention_mask = get_attention_pad_mask(dec_inputs, enc_inputs)

        dec_self_attentions = []
        for layer in self.layers:
            dec_outputs, dec_self_attention = layer(dec_outputs, dec_self_attention_mask)
            dec_self_attentions.append(dec_self_attention)
            # dec_enc_attentions.append(dec_enc_attention)
        return dec_outputs, dec_self_attentions


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attention = MultiHeadAttention()
        self.dec_enc_attention = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attention_mask, dec_enc_attention_mask):
        # 自注意力机制Q,K,V都是dec_inputs
        dec_outputs, dec_self_attention = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs,
                                                                  dec_self_attention_mask)
        # 这里用dec_outputs作为Q，enc_outputs作为K和V
        dec_outputs, dec_enc_attention = self.dec_enc_attention(dec_outputs, enc_outputs, enc_outputs,
                                                                dec_enc_attention_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attention, dec_enc_attention


class DecoderLayer2(nn.Module):
    def __init__(self):
        super(DecoderLayer2, self).__init__()
        self.dec_self_attention = MultiHeadAttention()
        # self.dec_enc_attention = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attention_mask):
        # 自注意力机制Q,K,V都是dec_inputs
        dec_outputs, dec_self_attention = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs,
                                                                  dec_self_attention_mask)
        # 这里用dec_outputs作为Q，enc_outputs作为K和V
        # dec_outputs, dec_enc_attention = self.dec_enc_attention(dec_outputs, enc_outputs, enc_outputs,
        # dec_enc_attention_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attention


def generate_rna_sequence(model, initial_input, length=20, temperature=1.0):
    model.eval()
    with torch.no_grad():
        current_input = initial_input

        for _ in range(length):
            # 前向传播
            current_input = current_input.to(device)
            output, _ = model(current_input)

            # 根据概率分布采样下一个字符
            next_char = torch.multinomial(torch.exp(output / temperature), 1)

            # 将下一个字符添加到序列中
            current_input = torch.cat([current_input[:, 1:], next_char], dim=1)

    return current_input


input_size = 640  # 输入特征的维度
hidden_size1 = 120  # 第一个隐含层的大小

length = 24181
tag_txt = 'sample_pro.txt'
tag_path = '/home2/public/data/RNA_aptamer/RNA-pro/' + tag_txt
directory_name = 'representations'
test_rate = 0.2
batch_size = 1

train_loader, test_loader = MyDataLoader(directory_name, tag_path, length, test_rate, batch_size)
# 加入transform
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = TransformerDecoderOnly(input_size, hidden_size1)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(10):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = torch.tensor(inputs).to(torch.long)
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs, dec_self_attns = model(inputs)
        # output:[batch_size x tgt_len,tgt_vocab_size]
        # 就这份代码而言，这里其实可以不写.contiguous()，因为target_batch这个tensor是连续的
        loss = criterion(outputs, labels)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

# for epoch in range(10):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs = torch.tensor(inputs).to(torch.long)
#         inputs = inputs.to(device)
#         labels = labels.float().to(device)
#         optimizer.zero_grad()
#
#         outputs, dec_self_attns = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#     print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
# model.eval()
# test_loss = 0.0
# for i, data in enumerate(test_loader, 0):
#     inputs, labels = data
#     inputs = torch.tensor(inputs).to(torch.long)
#     inputs = inputs.to(device)
#     labels = labels.float().to(device)
#     outputs, dec_self_attns = model(inputs)
#     loss = criterion(outputs, labels)
#     test_loss += loss.item()
#     test_loss = test_loss / len(test_loader)
# print(f' Testing Loss: {test_loss}')

model.eval()


def test():
    initial_input = np.load('/home2/public/data/RNA_aptamer/sample_embedding/representations/100.npy')
    torch_tensor = torch.from_numpy(initial_input)
    torch_tensor = torch_tensor.to(torch.long)
    torch_tensor = torch_tensor / 20
    generated_sequence = generate_rna_sequence(model, torch_tensor)
    print(generated_sequence)
    generated_sequence = generated_sequence.cpu()
    numpy_array = generated_sequence.numpy()
    np.savetxt('/home2/public/data/RNA_aptamer/100your_file.txt', numpy_array, fmt='%d')
    # 测试模型
#
#
# test()
#
# torch.save(model, '/home/sxtang/My_transformer/save/MyTransformModel.pth')
