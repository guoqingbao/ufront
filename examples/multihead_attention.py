import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param embed_dim: Size of each input sample.
        :param num_heads: Number of heads.
        :param dropout: Dropout rate
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('`embed_dim`({}) should be divisible by `num_heads`({})'.format(embed_dim, num_heads))
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.linear_q = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_o = nn.Linear(embed_dim, embed_dim, bias)
        self.dotproduct = ScaledDotProductAttention()
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        y = self.dotproduct(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return self.drop(y)

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'embed_dim={}, num_heads={}, dropout={}, bias={}, activation={}'.format(
            self.embed_dim, self.num_heads, self.dropout, self.bias, self.activation,
        )