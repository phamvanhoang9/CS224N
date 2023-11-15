import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init_(self, embd_size, enc_h_size, dec_h_size, v_size, device):
        super(EncoderRNN, self).__init__()
        self.enc_h_size = enc_h_size # enc_h_size is the size of hidden state of encoder
        self.dec_h_size = dec_h_size # dec_h_size is the size of hidden state of decoder
        self.v_size = v_size # v_size = vocabulary size
        self.device = device

        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU(embd_size, enc_h_size, bidirectional=True)
        self.f_concat_h = nn.Linear(enc_h_size*2, dec_h_size)

    def forward(self, x):
        # x: (T, batch_size, H), T = max_len
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded) # output: (T, batch_size, enc_h_size*2)
        hidden = torch.tanh(self.f_concat_h(torch.cat(hidden[-2], hidden[-1], dim=1))) # (1, batch_size, dec_h_size)
        # f_concat_h is a linear layer to transform the concatenated hidden state of forward and backward GRU
        # hidden[-2] is the last hidden state of forward GRU
        # hidden[-1] is the last hidden state of backward GRU
        # dim = 1 means concatenate along the second dimension
        # second dimension has size 2*enc_h_size

        return output, hidden.squeeze(0) # squeeze(0) means remove the first dimension, the first dimension size is 1
    
class Attention(nn.Module):
    def __init__(self, enc_h_size, dec_h_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_h_size*2 + dec_h_size, dec_h_size)
        """self.atten is a linear layer to transform the concatenated hidden state of encoder and decoder
        self.attn shape is (dec_h_size, enc_h_size*2 + dec_h_size)
        the purpose of self.attn is to transform the concatenated hidden state of encoder and decoder to a vector of size dec_h_size"""
        self.v = nn.Parameter(torch.rand(dec_h_size, 1, bias=False))
        """dec_h_size in rand means the size of the first dimension of the tensor
        1 in rand means the size of the second dimension of the tensor
        bias=False means no bias
        the purpose of self.v is to transform the vector of size dec_h_size to a scalar
        why do we need to transform the vector of size dec_h_size to a scalar?
        because we need to calculate the attention weight"""

    def forward(self, hidden, enc_outs):
        # hidden: (batch_size, dec_h_size)
        # enc_outs: (T, batch_size, enc_h_size*2)
        bs = enc_outs.shape[1] # batch_size
        src_len = enc_outs.shape[0]
        # repeat decoder hidden state src_len times
        dec_hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # (batch_size, src_len, dec_h_size)
        """hidden size is (batch_size, dec_h_size)
        hidden.unsqueeze(1) means add a dimension in the second dimension
        now hidden size is (batch_size, 1, dec_h_size)
        hidden.unsqueeze(1).repeat(1, src_len, 1) means repeat the tensor src_len times in the second dimension
        repeat(1, src_len, 1): the first dimension is not repeated, the second dimension is repeated src_len times, the third dimension is not repeated"""

        enc_outs = enc_outs.permute(1, 0, 2) # (batch_size, T, enc_h_size*2)
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, enc_outs), dim=2))) # (batch_size, src_len, enc_h_size*2+dec_h_size)
        """dim=2 means concatenate along the third dimension
        the third dimension is the dimension of dec_hidden and enc_outs
        I am curious why we defined attn above, but we use self.attn here
        I think we can use attn here, but we need to define attn in __init__ function
        Why do we use torch.cat((dec_hidden, enc_outs), dim=2) here?
        because we need to concatenate dec_hidden and enc_outs along the third dimension
        Why do we use torch.tanh here?
        because we need to transform the concatenated tensor to a tensor of the same size
        Is the torch.tanh is the tanh activation function?
        yes, torch.tanh is the tanh activation function
        In the tanh activation function formula, z is the concatenated tensor, so we need to concatenate dec_hidden and enc_outs"""
        attention = self.v(energy).squeeze(2) # (batch_size, src_len)
        """why do we need to use .squeeze(2) here?
        because we need to remove the third dimension
        But, why do we need to remove the third dimension?
        because we need to transform the tensor of size (batch_size, src_len, 1) to a tensor of size (batch_size, src_len)"""

        return F.softmax(attention, dim=1) # (batch_size, src_len) # dim=1 means softmax along the second dimension
        """Is src_len the sequeces length?
        yes, src_len is the sequeces length
        Does src_len contain the sequence encoded by encoder?
        yes, src_len contains the sequence encoded by encoder"""

class DecoderRNN(nn.Module):
    def __init__(self, embd_size, dec_h_size, v_size, device):
        super(DecoderRNN, self).__init__()
        self.dec_h_size = dec_h_size
        self.v_size = v_size
        self.device = device
        """I'm curious about why is the class EncoderRNN defined above needs dec_h_size? I think in the class EncoderRNN just needs enc_h_size, not dec_h_size
        I think the class EncoderRNN doesn't need dec_h_size, because the class EncoderRNN is used to encode the input sequence, not to decode the input sequence
        But above, we use dec_h_size in the class EncoderRNN, why?
        because we need to transform the concatenated hidden state of forward and backward GRU to a vector of size dec_h_size
        embd_size is the size of embedded equals to the size of the word vector
        v_size is the vocabulary size has the shape of (v_size, embd_size)"""
        
        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU(embd_size, dec_h_size)
        self.fout = nn.Linear(dec_h_size, v_size)
        """why is the fout size v_size?
        because the purpose of fout is to transform the hidden state of decoder to the vocabulary size
        why do we need the vocabulary size?
        because the purpose of decoder is to predict the next word, so we need the vocabulary size"""

    def forward(self, x, hidden, _enc_outs):
        """x is the input of decoder, x is the previous word
        hidden is the hidden state of decoder
        _enc_outs is the output of encoder
        why do we need _enc_outs here?
        because we need to calculate the attention weight
        x: (bs)
        hidden: (bs, H)
        _enc_outs: only used for AttnDecoderRNN and not used here.
        """
        x = x.unsqueeze(0) # (1, bs)
        hidden = hidden.unsqueeze(0) # (1, bs, H)
        """Why do we need to unqueeze x and hidden?
        because we need to add a dimension in the first dimension
        Why do we need to add a dimension in the first dimension?
        because the input of decoder is a word, so the size of the input is (1, bs)
        because the hidden state of decoder is a vector, so the size of the hidden state is (1, bs, H)"""
        embedded = self.embedding(x) # embedded size is (1, bs, embd_size)
        out, hidden = self.rnn(embedded, hidden) # out size is (1, bs, dec_h_size), hidden size is (1, bs, dec_h_size)
        """Why is the embedded_size dissapeared?
        because the input of rnn is embedded, so the input of rnn is (1, bs, embd_size)
        Is the dec_h_size as the embedded_size?
        no, dec_h_size is the size of hidden state of decoder, embedded_size is the size of embedded
        But, you said the size of embedded is (1, bs, embd_size)?
        yes, the size of embedded is (1, bs, embd_size), but the size of hidden state of decoder is (1, bs, dec_h_size)
        Is the out size (1, bs, dec_h_size)?
        yes, the out size is (1, bs, dec_h_size)"""
        out = self.fout(out.squeeze(0)) # (bs, v_size)
        out = F.softmax(out, dim=1) # (bs, v_size)
        """Is the v_size the vector?
        no, v_size is the vocabulary size
        What does the vocabulary size mean?
        the vocabulary size means the number of words in the vocabulary
        How does the vacabulary represent the words?
        the vocabulary represents the words by index
        Why do we need to use F.softmax here?
        because we need to calculate the probability of the next word"""
        
        return out, hidden.squeeze(0) # out: (bs, v_size), hidden: (bs, dec_h_size)
        """Why do we need to return out and hidden?
        because we need to use out to calculate the loss
        because we need to use hidden to calculate the next word"""

class AttnDecoderRNN(nn.Module):
    """What is the purpose of AttnDecoderRNN?
    the purpose of AttnDecoderRNN is to decode the input sequence with attention mechanism
    Why do we need attention mechanism?
    because we need to calculate the attention weight
    Why do we need to calculate the attention weight?
    because we need to calculate the context vector
    Why do we need to calculate the context vector?
    because we need to concatenate the context vector and the hidden state of decoder to predict the next word
    Why do we need to concatenate the context vector and the hidden state of decoder to predict the next word?
    because the purpose of decoder is to predict the next word
    Why do we need to predict the next word?
    because the purpose of decoder is to generate the output sequence
    Why do we need to generate the output sequence?
    because the purpose of seq2seq is to generate the output sequence"""
    def __init__(self, embd_size, enc_h_size, dec_h_size, v_size, attn, device):
        super(AttnDecoderRNN, self).__init__()
        self.enc_h_size = enc_h_size
        self.dec_h_size = dec_h_size
        self.v_size = v_size
        self.attn = attn
        self.device = device

        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU((embd_size + enc_h_size*2), dec_h_size)
        self.fout = nn.Linear(dec_h_size + enc_h_size*2 + embd_size, v_size)
        """What is the purpose of self.fout?
        the purpose of self.fout is to transform the concatenated hidden state of decoder, encoder and embedded to the vocabulary size
        """

    def forward(self, x, dec_hidden, enc_outs):
        """x is the input of decoder, x is the previous word
        dec_hidden is the hidden state of decoder
        enc_outs is the output of encoder
        x: (bs)
        dec_hidden: (bs, dec_h_size)
        enc_outs: (T, bs, enc_h_size*2)
        """
        x = x.unsqueeze(0) # (1, bs)
        embedded = self.embedding(x) # (1, bs, embd_size)
        """Why is the v_size replaced by (1, bs)?
        because the input of decoder is a word, so the size of the input is (1, bs)"""
        a = self.attn(dec_hidden, enc_outs).unsqueeze(1) # (bs, 1, src_len), a is the attention weight
        enc_outs = enc_outs.permute(1, 0, 2) # (bs, T, enc_h_size*2)
        weighted = torch.bmm(a, enc_outs) # (bs, 1, enc_h_size*2)
        weighted = weighted.permute(1, 0, 2) # (1, bs, enc_h_size*2)
        """What is the purpose of torch.bmm?
        the purpose of torch.bmm is to calculate the matrix multiplication of two tensors
        What is the purpose of torch.bmm(a, enc_outs)?
        the purpose of torch.bmm(a, enc_outs) is to calculate the context vector
        What is the context vector?
        the context vector is the weighted sum of the output of encoder
        Why do we need to calculate the context vector?
        because we need to concatenate the context vector and the hidden state of decoder to predict the next word
        Why do we need to concatenate the context vector and the hidden state of decoder to predict the next word?
        because the purpose of decoder is to predict the next word
        bmm is short for batch matrix multiplication"""
        rnn_input = torch.cat((embedded, weighted), dim=2)  # (1, bs, embd_size + enc_h_size*2)

        out, dec_hidden = self.rnn(rnn_input, dec_hidden.unsqueeze(0))  # out: (1, bs, dec_h_size), dec_hidden: (1, bs, dec_h_size)
        assert (out == dec_hidden).all()
        """Why do we need to use assert?
        because we need to check the condition is true or not
        Why do we need to check the condition is true or not?
        because we need to check the size of out and dec_hidden is the same
        """
        embedded = embedded.squeeze(0) # (bs, embd_size)
        out = out.squeeze(0) # (bs, dec_h_size)
        weighted = weighted.squeeze(0) # (bs, enc_h_size*2) 
        pred = self.fout(torch.cat((out, weighted, embedded), dim=1)) # (bs, v_size)

        return pred, dec_hidden.squeeze(0) # pred: (bs, v_size), dec_hidden: (bs, dec_h_size)
    
class Seq2Seq(nn.Module):
    """What is the purpose of Seq2Seq?
    the purpose of Seq2Seq is to generate the output sequence
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.enc_h_size == decoder.dec_h_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teaching_forcing_ratio=0.5):
        """
        src: (T, bs)
        trg: (T, bs), trg is the target sequence
        teaching_forcing_ratio: probability to use teaching forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        ground-truth inputs means the input is the previous word of target sequence
        """
        bs = src.shape[1] # batch_size, src size is (T, bs)
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.v_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, bs, trg_vocab_size).to(self.device)
        """Why do we need to use torch.zeros?
        because we need to initialize the outputs
        Why do we need to initialize the outputs?
        because we need to store the outputs
        outputs size is (trg_len, bs, trg_vocab_size)
        """

        enc_outs, hidden = self.encoder(src) # enc_outs size is (T, bs, enc_h_size*2), hidden size is (bs, dec_h_size)
        """Why do we need to use enc_outs and hidden?
        because we need to use enc_outs and hidden to initialize the decoder
        Why do we need to initialize the decoder?
        because we need to use the decoder to generate the output sequence
        """
        # first input to the decoder is the <sos> tokens
        inp = trg[0, :] # (bs)
        """Why do we need to use trg[0, :]?
        because the first input to the decoder is the <sos> tokens
        Why do we need to use trg[0, :] instead of trg[0]?
        because trg[0, :] means the first row of trg, trg[0] means the first column of trg
        """
        for t in range(1, trg_len):
            output, hidden = self.decoder(inp, hidden, enc_outs) # output size is (bs, trg_vocab_size), hidden size is (bs, dec_h_size)
            """why did the Decoder above define in the __init__ function contains the v_size? But, now using enc_outs
            """
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # get the highest predicted token from our predictions
            top1 = output.argmax(1) # top1 size is (bs)
            """Why argmax(1)?
            because we need to get the highest predicted token from our predictions
            How about argmax(0)?
            argmax(0) means get the highest predicted token from our predictions along the first dimension
            argmax(1) means get the highest predicted token from our predictions along the second dimension
            """
            teacher_force = random.random() < teaching_forcing_ratio
            """Why do we need to use random.random()?
            because we need to generate a random number between 0 and 1
            < teaching_forcing_ratio means the probability to use teaching forcing
            """
            inp = trg[t] if teacher_force else top1
            """if teacher_force is true, inp is trg[t]
            if teacher_force is false, inp is top1"""

        return outputs