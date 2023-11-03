#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
"""
from collections import namedtuple # https://docs.python.org/3/library/collections.html#collections.namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
"""
List: is a list of objects of the same type. 
Tuple: is a list of objects of different types.
Dict: is a dictionary, a list of key-value pairs.
Set: is a list of unique objects.
Union: is a list of objects of different types.
"""
import torch
import torch.nn as nn
import torch.nn.utils # https://pytorch.org/docs/stable/nn.html#torch-nn-utils
import torch.nn.functional as F # https://pytorch.org/docs/stable/nn.functional.html
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__() # initialize the super class (nn.Module)
        self.model_embeddings = ModelEmbeddings(embed_size, vocab) # initialize the model embeddings
        self.dropout_rate = dropout_rate # dropout rate
        self.vocab = vocab # vocabulary

        # default values
        self.encoder = None # encoder
        self.decoder = None # decoder
        self.h_projection = None # W_h
        self.c_projection = None # W_c
        self.att_projection = None # W_attProj
        self.combined_output_projection = None # W_u
        self.target_vocab_projection = None # W_vocab
        self.dropout = None
        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = False
        self.counter = 0 

        ### YOUR CODE HERE (~9 Lines)
        ### TODO - Initialize the following variables IN THIS ORDER:
        ###     self.post_embed_cnn (Conv1d layer with kernel size 2, input and output channels = embed_size,
        ###         padding = same to preserve output shape )
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        ###     Conv1D Layer:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        # Writing code here
        self.post_embed_cnn = nn.Conv1d(embed_size, embed_size, kernel_size=2, padding=1)
        """
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        @params:
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        """
        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size, bias=True)
        """
        What is the difference between nn.LSTM and nn.LSTMCell?
        Explanation:
        - nn.LSTMCell is designed to process a single sequence, whereas nn.LSTM is designed to process an entire batch of sequences.
        - nn.LSTMCell takes in the entire input sequence at once, whereas nn.LSTM takes in one input token at a time.
        - nn.LSTMCell returns only the hidden state, whereas nn.LSTM returns both the hidden state and the cell state.
        """
        self.h_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False) # shape (h, h), hidden_size = h
        """
        nn.Linear(in_features, out_features, bias=True)
        Why do we need to project the hidden state?
        Explanation:
        - The hidden state of the decoder is the input to the decoder LSTM cell.
        - The hidden state of the encoder is the output of the encoder LSTM cell.
        - The hidden state of the encoder and the hidden state of the decoder are of different dimensions.
        - The hidden state of the encoder is of dimension 2h, where h is the hidden size.
        - The hidden state of the decoder is of dimension h.
        - We need to project the hidden state of the encoder to the same dimension as the hidden state of the decoder.

        2 * hidden_size because we are using a bidirectional LSTM.
        nn.Linear(in_features, out_features, bias=True)
        """
        self.c_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        """
        What is the difference between self.h_projection and self.c_projection?
        Explanation:
        - self.h_projection is used to project the hidden state of the encoder.
        - self.c_projection is used to project the cell state of the encoder.

        So, please give me a definition of the hidden state and the cell state!
        Hidden state: The hidden state is the output of the LSTM cell. It is the output of the LSTM cell at the current time step. It is also the input to the LSTM cell at the next time step. 
        Cell state: The cell state is the memory of the LSTM cell. It is the memory of the LSTM cell at the current time step. It is also the memory of the LSTM cell at the next time step.
        """
        self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        """
        What is the attention projection layer?
        Attention projection layer is used to project the hidden states of the encoder to the same dimension as the hidden states of the decoder.
        """
        self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        """
        What is the combined output projection layer?
        The combined output projection layer is used to project the concatenation of the hidden state of the decoder and the attention output to the same dimension as the hidden state of the decoder.
        3 * hidden_size because we are concatenating the hidden state of the decoder and the attention output.
        attention output = attention scores * encoder hidden states
        """
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False) # shape (h, V), hidden_size = h, V is the size of the target vocabulary   
        """
        What is the target vocabulary projection layer?
        The target vocabulary projection layer is used to project the hidden state of the decoder to the same dimension as the target vocabulary.
        nn.Linear(in_features, out_features, bias=True)
        @params:
        in_features – size of each input sample has to be equal to the number of words in the source vocabulary
        out_features – size of each output sample has to be equal to the number of words in the target vocabulary
        """
        self.dropout = nn.Dropout(dropout_rate)
        ### END YOUR CODE

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source] # source_lengths: (b)

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths) # enc_hiddens: (b, src_len, h*2), dec_init_state: (h, c)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths) # enc_masks: (b, src_len)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded) # combined_outputs: (tgt_len, b, h)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1) # P: (tgt_len, b, V), dim=-1 means the last dimension. V is the size of the target vocabulary. 
        # We need the last dim because we want to compute the log probability distribution over the target vocabulary. What is the last dim? The last dim is the dimension of the target vocabulary.
        """
        P looks like this:
        P = [
            [
                [p_1_1_1, p_1_1_2, ..., p_1_1_V],
                [p_1_2_1, p_1_2_2, ..., p_1_2_V],
                ...
                [p_1_b_1, p_1_b_2, ..., p_1_b_V]
            ],
            [
                [p_2_1_1, p_2_1_2, ..., p_2_1_V],
                [p_2_2_1, p_2_2_2, ..., p_2_2_V],
                ...
                [p_2_b_1, p_2_b_2, ..., p_2_b_V]
            ],
            ...
            [
                [p_tgt_len_1_1, p_tgt_len_1_2, ..., p_tgt_len_1_V],
                [p_tgt_len_2_1, p_tgt_len_2_2, ..., p_tgt_len_2_V],
                ...
                [p_tgt_len_b_1, p_tgt_len_b_2, ..., p_tgt_len_b_V]
            ]
        ]
        """
        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:] # P: (tgt_len, b, V), target_padded[1:]: (tgt_len-1, b), target_masks[1:]: (tgt_len-1, b), target_gold_words_log_prob: (tgt_len-1, b)
        """
        index=target_padded[1:] means that we want to compute the log probability of generating the gold-standard target sentence for each example in the input batch.
        Why target_padded[1:]? Because we want to exclude the <s> token. What about </s> token? We don't need to exclude the </s> token because we have already excluded the <pad> token.
        unsqueeze(-1) means that we want to add a dimension at the end of the tensor. Why do we want to add a dimension at the end of the tensor? Because we want to use the gather function.
        Why gather function need a dimension at the end of the tensor? Because gather function needs to know which dimension to gather from.
        dim=-1 means that we want to gather from the last dimension. Last dimension means the dimension at the end of the tensor, e.g., the dimension of V.
        squeeze(-1) means that we want to remove the dimension at the end of the tensor. Why do we want to remove the dimension at the end of the tensor? Because we want to compute the log probability of generating the gold-standard target sentence for each example in the input batch.
        """
        scores = target_gold_words_log_prob.sum(dim=0) # scores: (b, ), dim=0 means the first dimension. We need the first dim because we want to compute the log-likelihood of generating the gold-standard target sentence for each example in the input batch. Here b = batch size.
        """
        Why do we need to sum over the first dimension? Because we want to compute the log-likelihood of generating the gold-standard target sentence for each example in the input batch.
        sum over the first dimension means that we want to sum over the first dimension of the tensor.
        """
        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: # Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] means (enc_hiddens, dec_init_state)
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell. Both tensors should have shape (2, b, h).
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 11 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the encoder.
        ###     2. Apply the post_embed_cnn layer. Before feeding X into the CNN, first use torch.permute to change the
        ###         shape of X to (b, e, src_len). After getting the output from the CNN, still stored in the X variable,
        ###         remember to use torch.permute again to revert X back to its original shape.
        ###     3. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor output returned by the encoder RNN is (src_len, b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`, so you may need to do more permuting.
        ###         - Note on using pad_packed_sequence -> For batched inputs, you need to make sure that each of the
        ###           individual input examples has the same shape.
        ###     4. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/generated/torch.cat.html
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/generated/torch.permute.html
        ###     Tensor Reshape (a possible alternative to permute):
        ###         https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html

        # Writing code here
        X = self.model_embeddings.source(source_padded) # X: (src_len, b, e), source_padded: (src_len, b), e = embedding size
        X = X.permute(1, 2, 0) # X: (b, e, src_len)
        """
        Why should we permute X?
        Answer:
        - Because the input to the CNN should be of shape (b, e, src_len).
        - But the shape of X is (src_len, b, e).
        - So, we need to permute X.
        - Documentation: https://pytorch.org/docs/stable/generated/torch.Tensor.permute.html
        I don't know why CNN needs the input to be of shape (b, e, src_len)?
        Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        X = self.post_embed_cnn(X) # X: (b, e, src_len)
        X = X.permute(2, 0, 1) # X: (src_len, b, e)
        """
        Why should we permute X?
        Answer:
        - Because the output of the CNN should be of shape (src_len, b, e).
        """
        X = pack_padded_sequence(X, torch.LongTensor(source_lengths)) # X: (src_len, b, e)

        enc_hiddens, (last_hidden, last_cell) = self.encoder(X) # enc_hiddens: (src_len, b, h*2), last_hidden: (2, b, h), last_cell: (2, b, h)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens) # enc_hiddens: (src_len, b, h*2)
        enc_hiddens = enc_hiddens.permute(1, 0, 2) # enc_hiddens: (b, src_len, h*2)

        init_decoder_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1) # init_decoder_hidden: (b, 2*h)
        """
        Why should we concatenate the forwards and backwards tensors?
        last_hidden is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        dim=1 means that we want to concatenate the forwards and backwards tensors along the second dimension. The second dimension corresponds to the batch size.
        """
        init_decoder_hidden = self.h_projection(init_decoder_hidden) # init_decoder_hidden: (b, h)
        init_decoder_cell = torch.cat((last_cell[0], last_cell[1]), dim=1) # init_decoder_cell: (b, 2*h)
        init_decoder_cell = self.c_projection(init_decoder_cell) # init_decoder_cell: (b, h)
        dec_init_state = (init_decoder_hidden, init_decoder_cell) # dec_init_state: (h, c)

        ### END YOUR CODE

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop off the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0) # enc_hiddens.size(0) means the first dimension of enc_hiddens tensor. The first dimension of enc_hiddens tensor is the batch size.
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device) # o_prev: (b, h)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e).
        ###             - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### You may find some of these functions useful:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack

        # Writing code here
        enc_hiddens_proj = self.att_projection(enc_hiddens) # enc_hiddens_proj: (b, src_len, h)
        Y = self.model_embeddings.target(target_padded) # Y: (tgt_len, b, e), target_padded: (tgt_len, b), e = embedding size
        for Y_t in torch.split(Y, 1): # Y_t: (1, b, e)
            Y_t = torch.squeeze(Y_t, dim=0)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        combined_outputs = torch.stack(combined_outputs) # combined_outputs: (tgt_len, b, h)

        ### END YOUR CODE

        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t (be careful about the input/ output shapes!)
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze


        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        # $$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh

        # Writing code here
        dec_state, _ = self.decoder(Ybar_t, dec_state) # dec_state: (b, h), dec_hidden: (b, h), dec_cell: (b, h)
        dec_hidden, dec_cell = dec_state # dec_hidden: (b, h), dec_cell: (b, h)
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2) # e_t: (b, src_len)
        alpha_t = F.softmax(e_t, dim=1) # alpha_t: (b, src_len)
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)
        U_t = torch.cat((dec_hidden, a_t), dim=1) # U_t: (b, 2*h)
        V_t = self.combined_output_projection(U_t) # V_t: (b, h)
        O_t = self.dropout(torch.tanh(V_t)) # O_t: (b, h)

        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode='floor')
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
