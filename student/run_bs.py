import argparse # for command line parsing
import math
import os
import time

import spacy # for tokenization
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k # Multi30k dataset
from torchtext.data import Field, BucketIterator # Field: how data should be processed, BucketIterator: to get batches of data

from beam import beam_search_decoding, batch_beam_search_decoding
from model import EncoderRNN, DecoderRNN, Attention, AttnDecoderRNN, Seq2Seq


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1] # reverse the order of the tokens

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def init_weights(m):
    for name, param in m.named_parameter():
        nn.init.uniform_(param.data, -0.08, 0.08)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time # in seconds
    elapsed_mins = int(elapsed_time/60) # in minutes
    elapsed_secs = int(elapsed_time - (elapsed_mins*60)) # in seconds
    return elapsed_mins, elapsed_secs

def print_n_best(decoded_seq, itos):
    for rank, seq in enumerate(decoded_seq):
        print(f'Out: Rank-{rank+1}: {" ".join([itos[idx] for idx in seq])}')

def train(model, itr, optimizer, criterion):
    print('Start training...')
    model.train()
    epoch_loss = 0
    for batch in itr:
        src = batch.src # [src_len, batch_size]
        trg = batch.trg # [trg_len, batch_size]

        optimizer.zero_grad()

        output = model(src, trg) # [trg_len, batch_size, output_dim]

        output_size = output.shape[-1]

        output = output[1:].view(-1, output_size) 
        """
        output[1:] means we ignore the first token of each sequence (which is <sos>)
        .view(-1, output_size) means we flatten each of the predictions"""
        trg = trg[1:].view(-1) # [trg_len * batch_size]

        loss = criterion(output, trg) 
        """what is the criterion?
        The loss function calculates the average loss per token, however by passing the index of the <pad> token as the ignore_index argument
        we ignore the loss whenever the target token is a padding token."""
        loss.backward() # backpropagate the loss
        optimizer.step() # update the parameters

        epoch_loss += loss.item() # loss.item() returns the average loss of the batch

    return epoch_loss / len(itr)

def evaluate(model, itr, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in itr:
            src = batch.src 
            trg = batch.trg

            output = model(src, trg, 0) # turn off teacher forcing

            output_dim = output.shape[-1] # output_dim = output.shape[2]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(itr)

DEVICE = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')

def main():
    parser = argparse.ArgumentParser()

    # hyper parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--enc_emb_dim', type=int, default=256)
    parser.add_argument('--dec_emb_dim', type=int, default=256)
    parser.add_argument('--enc_hid_dim', type=int, default=512)
    parser.add_argument('--dec_hid_dim', type=int, default=512)

    # other parameters
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--n_best', type=int, default=5)
    parser.add_argument('--max_dec_steps', type=int, default=1000)
    parser.add_argument('--export_dir', type=str, default='./ckpts/')
    parser.add_argument('--model_name', type=str, default='s2s')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--attention', action='store_true') # if we want to use attention mechanism or not (default: False)
    # how can I set the attention mechanism to be true when I run the code?
    # python run_bs.py --attention True --model_path ./ckpts/s2s.pt --skip_train True --beam_size 10 --n_best 5 --max_dec_steps 1000 --export_dir ./ckpts/ --model_name s2s --batch_size 128 --epochs 10 --enc_emb_dim 256 --dec_emb_dim 256 --enc_hid_dim 512 --dec_hid_dim 512 

    opts = parser.parse_args()


    SOS_token = '<SOS>'
    EOS_token = '<EOS>'

    SRC = Field(tokenize=tokenize_de,
                init_token=SOS_token,
                eos_token=EOS_token,
                lower=True)
    TRG = Field(tokenize=tokenize_en,
                init_token=SOS_token,
                eos_token=EOS_token,
                lowe=True)
    
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    print(f'Number of training examples: {len(train_data.examples)}')
    print(f'Number of validation examples: {len(valid_data.examples)}')
    print(f'Number of testing examples: {len(test_data.examples)}')

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    print(f'Unique tokens in source (de) vocabulary: {len(SRC.vocab)}')
    print(f'Unique tokens in target (en) vocabulary: {len(TRG.vocab)}')

    train_itr, valid_itr, test_itr =\
            BucketIterator.splits(
                (train_data, valid_data, test_data),
                batch_size=opts.batch_size,
                device=DEVICE
            )
    
    enc_v_size = len(SRC.vocab)
    dec_v_size = len(TRG.vocab)

    encoder = EncoderRNN(opts.enc_embd_dim, opts.enc_hid_dim, opts.dec_hid_dim, enc_v_size, DEVICE)
    if opts.attention:
        attn = Attention(opts.enc_hid_dim, opts.dec_hid_dim)
        decoder = AttnDecoderRNN(opts.dec_emb_dim, opts.enc_hid_dim, opts.dec_hid_dim, dec_v_size, attn, DEVICE)
    else:
        decoder = DecoderRNN(opts.dec_emb_dim, opts.enc_hid_dim, opts.dec_hid_dim, dec_v_size, DEVICE)
    
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    """Why we need to add th DEVICE twice?
    Because the encoder and decoder are also on the DEVICE"""

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token] # get the index of the <pad> token

    if opts.model_path != '':
        model.load_state_dict(torch.load(opts.model_path))
    
    if not opts.skip_train:
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
        best_valid_loss = float('inf')
        for epoch in range(opts.epochs):
            start_time = time.time()
            train_loss = train(model, train_itr, optimizer, criterion)
            valid_loss = evaluate(model, valid_itr, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                attn_type = 'attn' if opts.attention else 'vanilla'
                model_path = os.path.join(opts.export_dir, f'{opts.model_name}-{attn_type}.pt')
                print(f'Update model! Saved at {model_path}')
                torch.save(model.state_dict(), model_path)
            else:
                print('Model was not updated. Stop training')
                break

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s') # +1:02 means we want to print the epoch number with 2 digits
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\tVal Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f}')

    TRG_SOS_IDX = TRG.vocab.stoi[TRG.init_token]
    TRG_EOS_IDX = TRG.vocab.stoi[TRG.eos_token]
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(test_itr):
            src = batch.src # [src_len, batch_size]
            trg = batch.trg # [trg_len, batch_size]
            print(f'In: {" ".join(SRC.vocab.itos[idx] for idx in src[:,0])}')

            enc_outs, h = model.encoder(src) # (T, bs, H), (bs, H)
            start_time = time.time()
            decoded_seqs = beam_search_decoding(
                decoder=model.decoder,
                enc_outs=enc_outs,
                enc_last_h=h,
                beam_width=opts.beam_size,
                n_best=opts.n_best,
                sos_token=TRG_SOS_IDX,
                eos_token=TRG_EOS_IDX,
                max_dec_steps=opts.max_dec_steps,
                device=DEVICE
            )

            end_time = time.time()
            print(f'Beam search decoding time: {end_time-start_time:.3f}s')
            print_n_best(decoded_seqs[0], TRG.vocab.itos)
            
            start_time = time.time()
            decoded_seqs = batch_beam_search_decoding(
                decoder=model.decoder,
                enc_outs=enc_outs,
                enc_last_h=h,
                beam_width=opts.beam_size,
                n_best=opts.n_best,
                sos_token=TRG_SOS_IDX,
                eos_token=TRG_EOS_IDX,
                max_dec_steps=opts.max_dec_steps,
                device=DEVICE
            )
            end_time = time.time()
            print(f'Batch beam search decoding time: {end_time-start_time:.3f}s')
            print_n_best(decoded_seqs[0], TRG.vocab.itos)

        
if __name__ == '__main__':
    main()