import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from math import sqrt

# BertConfig {
#   "_name_or_path": "bert-base-uncased",
#   "architectures": [
#     "BertForMaskedLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "classifier_dropout": null,
#   "gradient_checkpointing": false,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-12,
#   "max_position_embeddings": 512,
#   "model_type": "bert",
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   "pad_token_id": 0,
#   "position_embedding_type": "absolute",
#   "transformers_version": "4.42.4",
#   "type_vocab_size": 2,
#   "use_cache": true,
#   "vocab_size": 30522
# }

def get_input_embed():
    # text -> token embeddings
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    print('input ids : ', inputs.input_ids)

    config = AutoConfig.from_pretrained(model_ckpt)
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    print('token embedding : ', token_emb) # [30522, 768]

    inputs_embeds = token_emb(inputs.input_ids)
    print('inputs embedings : ', inputs_embeds.size()) # [1, 5, 768] [batch_size, seq_len, hidden_dim]
    return inputs_embeds, config

def attention_example():
    print('getting inputs : ')
    inputs_embeds, config = get_input_embed() # [1, 5, 768] [batch_size, seq_len, hidden_dim]
    # query key value -> attention
    Q = K = V = inputs_embeds
    dim_k = K.size(-1)

    print('getting outputs : ')
    scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(dim_k)
    print('socres size : ', scores.size()) # [1, 5, 5]

    weights = F.softmax(scores, dim=-1) # [1, 5, 5]
    print('weights : ', weights)
    print('weights : ', weights.sum(dim=-1))

    attn_outputs = torch.bmm(weights, V)
    print('attention outputs : ', attn_outputs.shape) # [1, 5, 768] [batch_size, seq_len, hidden_dim]
    print()

# print('------------ test attention example ------------------')
# attention_example()

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float('inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim): # head_dim = embed_dim / num_heads
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim) # batch_size, seg_len, head_dim
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    

def attn_func_test():
    print('getting inputs : ')
    inputs_embeds, config = get_input_embed() # [1, 5, 768] [batch_size, seq_len, hidden_dim]

    # inputs embedings -> attention
    multihead_attn = MultiHeadAttention(config)
    query = key = value = inputs_embeds

    print('getting outputs : ')
    attn_outputs = multihead_attn(query, key, value)
    # print(attn_outputs)
    print('attn outputs : ', attn_outputs.size()) # [1, 5, 768] [batch_size, seq_len, hidden_dim]

    # feed forward
    feed_forward = FeedForward(config)
    ff_outputs = feed_forward(attn_outputs)
    print('ff outputs : ', ff_outputs.size()) # [1, 5, 768]
    print()

# print('------------ test multi head attention ------------------')
# attn_func_test()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x, mask=None):
        # Here we use pre layer normalization, LN + Attn and LN + FF
        # apply Layer Normalization and then copy input into query , key, value
        hidden_state = self.layer_norm_1(x)
        
        # apply attention with a skip-connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        
        # apply feed-forward layer with a skip-connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

def test_transformer_enc_layer():
    print('getting inputs : ')
    inputs_embeds, config = get_input_embed() # [1, 5, 768] [batch_size, seq_len, hidden_dim]
    encoder_layer = TransformerEncoderLayer(config)
    
    print('getting outputs : ')
    outputs = encoder_layer(inputs_embeds)
    print('outputs : ', outputs.shape)

# print('------------ test transformer encoder layer ------------------')
# test_transformer_enc_layer()


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()
    
    def forward(self, input_ids):
        # create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        
        # create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def test_embeddings():
    # text -> token embeddings
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    print('input ids : ', inputs.input_ids)

    config = AutoConfig.from_pretrained(model_ckpt)

    embedding_layer = Embeddings(config)
    embeddings = embedding_layer(inputs.input_ids)
    print('embeddings : ', embeddings.shape) 


# print('------------ test embeddings ------------------')
# test_embeddings()

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


def test_transfomer_encoder():
    # text -> token embeddings
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    print('input ids : ', inputs.input_ids)

    config = AutoConfig.from_pretrained(model_ckpt)

    encoder = TransformerEncoder(config)
    outputs = encoder(inputs.input_ids)
    print('outputs : ', outputs.shape)


# print('------------ test transformer encoder ------------------')
# test_transfomer_encoder()


