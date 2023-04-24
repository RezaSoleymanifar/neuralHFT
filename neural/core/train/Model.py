import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



#-----------------------------------------------------------------------------------------------
class gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x,3))))


class gelu2(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


#-----------------------------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = config.embedding
        self.n_head = config.n_head
        assert self.embedding % self.n_head==0
        self.embedding_head = config.embedding // config.n_head
        self.Wq = nn.Linear(self.embedding, self.embedding, bias = False)
        self.Wk = nn.Linear(self.embedding, self.embedding, bias = False)
        self.Wv = nn.Linear(self.embedding, self.embedding, bias = False)
        self.project =  nn.Linear(self.embedding, self.embedding, bias = False)
 
    def _split_head(self, tensor):
        batch, seq_len, embbeding = tensor.size()
        split = tensor.view(batch, seq_len, self.n_head,  self.embedding_head)
        output = split.permute(0,2,1,3)
        return output


    def mask_attn_weights(self, score_unormalize):
        _, _, nd,ns = score_unormalize.shape
        mask_matrice = torch.tril(torch.ones(nd,ns),ns - nd)
        out = score_unormalize*mask_matrice - 1e20*(1 - mask_matrice)
        return out

    def forward(self, input_SA, atten_mask = True):

        batch, seq_len, input_dim = input_SA.size()
        assert self.embedding == input_dim
        q = self.Wq(input_SA)  #(batch, seq_len, embedding)
        k = self.Wk(input_SA)  #(batch, seq_len, embedding)
        v = self.Wv(input_SA)  #(batch, seq_len, embedding)

        q = self._split_head(q) #(batch, n_head, seq_len, embedding_head)
        k = self._split_head(k) #(batch, n_head, seq_len, embedding_head)
        v = self._split_head(v) #(batch, n_head, seq_len, embedding_head)
        present = torch.stack([k, v], 1)

        score_unormalize = torch.matmul(q, k.transpose(-1, -2))  # (batch, n_head, seq_len, seq_len)
        if atten_mask:
            score_unormalize = self.mask_attn_weights(score_unormalize)


        attn_normalize = F.softmax(score_unormalize, -1)/ math.sqrt(self.embedding_head)
        res = torch.matmul(attn_normalize, v)
        result = res.transpose(1, 2).contiguous().view(batch, seq_len, self.embedding)  #(batch, seq_len, embedding)
        output = self.project(result)
        return (output, attn_normalize, present)

#-----------------------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.embedding
        intermediate = config.mlp_intermediate
        self.h = nn.Linear(input_dim, intermediate)
        self.out = nn.Linear(intermediate, input_dim)
        self.act = config.ACT()

    def forward(self, input_mlp):
        l1 = self.act(self.h(input_mlp))
        return self.out(l1)

#-----------------------------------------------------------------------------------------------

class block(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding = config.embedding
        self.LN1 = nn.LayerNorm(embedding)
        self.sa = SelfAttention(config)
        self.LN2 = nn.LayerNorm(embedding)
        self.mlp = MLP(config)

    def forward(self, x):
        attn, *extra = self.sa(self.LN1(x))
        res1 =  x + attn
        res2 =  res1 + self.mlp(self.LN2(res1))
        return res2, extra

#-----------------------------------------------------------------------------------------------

class LM_engine(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config.vocab_size
        self.wte = nn.Embedding(vocab_size, config.embedding)
        self.pe = nn.Embedding(vocab_size, config.embedding)
        self.h = nn.ModuleList([block(config) for i in range(config.n_layers)])
    def forward(self, input_ids):
        h = self.wte(input_ids)
        present_buffer = []
        for layer in self.h:
            h, *extra = layer(h)
            present_buffer.append(extra[-1])
        out = h
        return out
        
#-----------------------------------------------------------------------------------------------

class LM(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config.vocab_size
        self.lm_engine = LM_engine(config)
        self.project = nn.Linear(config.embedding, vocab_size, bias = False)
    def forward(self, input_ids):
        last_layer = self.lm_engine(input_ids)
       
        out = self.project(last_layer)
        return out
        

        



