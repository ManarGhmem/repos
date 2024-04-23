# import torch
# import torch.nn as nn


# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size) 
#         self.l2 = nn.Linear(hidden_size, hidden_size) 
#         self.l3 = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         out = self.relu(out)
#         out = self.l3(out)
#         # no activation and no softmax at the end
#         return out








# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.dropout1 = nn.Dropout(0.5)
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#         self.dropout2 = nn.Dropout(0.5)
#         self.l3 = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.l1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout1(out)
#         out = self.l2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.dropout2(out)
#         out = self.l3(out)
#         return out


# Model
import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def SelfAttention(q , k , v , mask) :
    # attention weights
    attention_logits = torch.matmul(q , k.transpose(-2 , -1)).to(DEVICE)

    # number of embedding values
    scaling = torch.sqrt(torch.tensor(k.size(-1) , dtype = torch.float32)).to(DEVICE)

    # scaled attention weights
    scaled_attention_logits = attention_logits / scaling

    # Apply mask to the attention weights (if mask is not None)
    if mask is not None :
        scaled_attention_logits += (mask * -1e9)

    # apply softmax
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1).to(DEVICE)

    # Compute the weighted sum of the value vectors using the attention weights
    output = torch.matmul(attention_weights , v).to(DEVICE)

    return output

class MultiHeadAttention(nn.Module) :
    def __init__(self , embedding_dim , num_heads) :
        super(MultiHeadAttention , self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Embedding size needs to be divisible by heads
        assert self.embedding_dim % self.num_heads == 0

        self.head_dim = self.embedding_dim // self.num_heads

        self.queries = nn.Linear(embedding_dim , embedding_dim)
        self.keys = nn.Linear(embedding_dim , embedding_dim)
        self.values = nn.Linear(embedding_dim , embedding_dim)
        self.fc_out = nn.Linear(embedding_dim , embedding_dim)

    def split_heads(self , x , batch_size) :
        x = x.reshape(batch_size , -1 , self.num_heads , self.head_dim)
        return x.permute(0, 2, 1, 3) # (batch_size , num_heads , seqlen , head_dim)

    def forward(self , q , k , v , mask = None) :
        batch_size = q.shape[0]

        # (q , k , v) shape: (batch_size , seqlen , embedding_dim)
        q = self.queries(q)
        k = self.keys(k)
        v = self.values(v)

        q = self.split_heads(q , batch_size)
        k = self.split_heads(k , batch_size)
        v = self.split_heads(v , batch_size)

        # scaled_attention shape : (batch_size , num_heads , seqlen_q , head_dim)
        scaled_attention = SelfAttention(q , k , v , mask)

        scaled_attention = scaled_attention.permute(0 , 2 , 1 , 3) # (batch_size  , seqlen_q , num_heads , head_dim)

        # Concatenation of heads
        attention_output = scaled_attention.reshape(batch_size, -1, self.embedding_dim)  # (batch_size, seq_len_q, embedding_dim)

        out = self.fc_out(attention_output) # (batch_size , seqlen_q , embedding_dim)

        return out

class EncoderBlock(nn.Module) :
    def __init__(self , embedding_dim , num_heads , fc_dim , dropout_rate = 0.1) :
        super(EncoderBlock , self).__init__()

        self.MHA = MultiHeadAttention(embedding_dim , num_heads)

        self.norm1 = nn.LayerNorm(embedding_dim , eps=1e-6)
        self.norm2 = nn.LayerNorm(embedding_dim , eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim , fc_dim) ,
            nn.ReLU() ,
            nn.Linear(fc_dim , embedding_dim)
        )

    def forward(self , x , mask) :
        attn_out = self.MHA(x , x , x , mask)
        attn_out = self.dropout1(attn_out)

        out1 = self.norm1(x + attn_out)

        fc_out = self.dropout2(self.fc(out1))

        enc_out = self.norm2(out1 + fc_out)

        return enc_out # (batch_size , seqlen , embedding_dim)

class Encoder(nn.Module) :
    def __init__(
        self ,
        num_layers ,
        embedding_dim ,
        num_heads ,
        fc_dim ,
        src_vocab_size ,
        max_length ,
        dropout_rate = 0.1
    ) :
        super(Encoder , self).__init__()

        self.num_layers = num_layers

        self.embedding = nn.Embedding(src_vocab_size , embedding_dim)

        self.pos_encoding = nn.Embedding(max_length , embedding_dim)

        self.enc_layers = [EncoderBlock(embedding_dim , num_heads , fc_dim , dropout_rate).to(DEVICE)
                          for _ in range(num_layers)]

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self , x , mask) :
        batch_size , seqlen = x.shape

        positions = torch.arange(0, seqlen).expand(batch_size, seqlen).to(DEVICE)

        out = self.dropout((self.embedding(x) + self.pos_encoding(positions)))

        for i in range(self.num_layers) :
            out = self.enc_layers[i](out , mask)

        return out # (batch_size , seqlen , embedding_dim)

class DecoderBlock(nn.Module) :
    def __init__(self , embedding_dim , num_heads , fc_dim , dropout_rate = 0.1) :
        super(DecoderBlock , self).__init__()

        self.MHA1 = MultiHeadAttention(embedding_dim , num_heads)
        self.MHA2 = MultiHeadAttention(embedding_dim , num_heads)

        self.norm1 = nn.LayerNorm(embedding_dim , eps=1e-6)
        self.norm2 = nn.LayerNorm(embedding_dim , eps=1e-6)
        self.norm3 = nn.LayerNorm(embedding_dim , eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim , fc_dim) ,
            nn.ReLU() ,
            nn.Linear(fc_dim , embedding_dim)
        )


    def forward(self , x , enc_output , look_ahead_mask , padding_mask) :
        # enc_output shape : (batch_size , seqlen , embedding_dim)

        attn1 = self.MHA1(x , x , x , look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.norm1(attn1 + x)

        attn2 = self.MHA2(out1 , enc_output , enc_output , padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.norm2(attn2 + out1)

        fc_out = self.dropout3(self.fc(out2))
        dec_out = self.norm3(fc_out + out2)

        return dec_out # (batch_size , seqlen , embedding_dim)

class Decoder(nn.Module) :
    def __init__(
        self ,
        num_layers ,
        embedding_dim ,
        num_heads ,
        fc_dim ,
        trg_vocab_size ,
        max_length ,
        dropout_rate = 0.1
    ) :
        super(Decoder , self).__init__()

        self.num_layers = num_layers

        self.embedding = nn.Embedding(trg_vocab_size , embedding_dim)

        self.pos_encoding = nn.Embedding(max_length , embedding_dim)

        self.dec_layers = [DecoderBlock(embedding_dim , num_heads , fc_dim , dropout_rate).to(DEVICE)
                          for _ in range(num_layers)]

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self , x , enc_output , look_ahead_mask , padding_mask) :
        batch_size , seqlen = x.shape

        positions = torch.arange(0, seqlen).expand(batch_size, seqlen).to(DEVICE)

        out = self.dropout((self.embedding(x) + self.pos_encoding(positions)))

        for i in range(self.num_layers) :
            out  = self.dec_layers[i](out , enc_output , look_ahead_mask , padding_mask)


        return out # (batch_size , seqlen , embedding_dim)

class Transformer(nn.Module) :
    def __init__(
        self ,
        num_layers ,
        embedding_dim ,
        num_heads ,
        fc_dim ,
        src_vocab_size ,
        trg_vocab_size ,
        src_max_length ,
        trg_max_length ,
        dropout_rate = 0.1
    ) :
        super(Transformer , self).__init__()

        self.encoder = Encoder(
            num_layers ,
            embedding_dim ,
            num_heads ,
            fc_dim ,
            src_vocab_size ,
            src_max_length ,
            dropout_rate
        ).to(DEVICE)

        self.decoder = Decoder(
            num_layers ,
            embedding_dim ,
            num_heads ,
            fc_dim ,
            trg_vocab_size ,
            trg_max_length ,
            dropout_rate
        ).to(DEVICE)

        self.fc_out = nn.Linear(embedding_dim , trg_vocab_size)

    def padding_mask(self , seq) :
        # Mask all the pad tokens in the batch of sequence.
        # It ensures that the model does not treat padding as the input.
        #The mask indicates where pad value 0 is present: it outputs a 1 at those locations,and a 0 otherwise.

        # seq shape : (batch_size , seqlen)  -> (batch_size , 1 , 1 , seq_len)

        seq_mask = (seq == 0).float().unsqueeze(1).unsqueeze(2)

        return seq_mask


    def look_ahead_mask(self , trg) :
        # The look-ahead mask is used to mask the future tokens in a sequence. In other words,
        # the mask indicates which entries should not be used

        # Returns a lower triangular matrix filled with 1s. The shape of the mask is (target_size, target_size)
        # tensor([[[[0., 1., 1.],
                  #  [0., 0., 1.],
                  #  [0., 0., 0.]]]])

        batch_size , trg_len = trg.shape

        trg_mask = 1 - torch.tril(torch.ones((trg_len , trg_len)) , diagonal=0).expand(
            batch_size, 1, trg_len, trg_len
        )

        return trg_mask

    def create_masks(self , src , trg) :
        # encoder padding mask
        enc_padding_mask = self.padding_mask(src).to(DEVICE)

        # decoder padding mask
        dec_padding_mask = self.padding_mask(src).to(DEVICE)

        look_ahead_mask = self.look_ahead_mask(trg).to(DEVICE)

        dec_trg_padding_mask = self.padding_mask(trg).to(DEVICE)

        combined_mask = torch.max(dec_trg_padding_mask , look_ahead_mask).to(DEVICE)

        return enc_padding_mask , combined_mask , dec_padding_mask

    def forward(self , src , trg) :

        enc_padding_mask , look_ahead_mask , dec_padding_mask = self.create_masks(src , trg)

        enc_output = self.encoder(src , enc_padding_mask)

        dec_output = self.decoder(trg , enc_output , look_ahead_mask, dec_padding_mask)

        out = self.fc_out(dec_output)

        return out
