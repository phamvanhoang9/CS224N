import copy # for deepcopy
from heapq import heappush, heappop # for priority queue
import torch


class BeamSearchNode(object):
    def __init__(self, h, prev_node, wid, logp, length):
        """
        @param h: decoder hidden state
        @param prev_node: previous decoder state
        @param wid: word id
        @param logp: log probability of the word
        @param length: length of the sequence
        """
        self.h = h
        self.prev_node = prev_node
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6) # 1e-6 to prevent log(0)
    
def beam_search_decoding(
        decoder,
        enc_outs,
        enc_last_h,
        beam_width,
        n_best,
        sos_token,
        eos_token,
        max_dec_steps,
        device,
        ):
    """
    @param decoder: decoder model
    @param enc_outs: encoder outputs, shape = (T, B, 2*H)
    @param enc_last_h: encoder last hidden state, shape = (B, H)
    @param beam_width: beam width
    @param n_best: n best sequences
    @param sos_token: start of sentence token
    @param eos_token: end of sentence token
    @param max_dec_steps: maximum decoding steps
    @param device: device type

    Returns:
    @param n_best_list: list of n best sequences
    """

    assert beam_width >= n_best

    n_best_list = []
    bs = enc_outs.shape[1] # batch size

    for batch_id in range(bs):
        # Get last encoder hidden state
        decoder_hidden = enc_last_h[batch_id] # (H)
        enc_out = enc_outs[:, batch_id].unsqueeze(1) # (T, 1, 2*H)

        # Prepare first token for decoder
        decoder_input = torch.tensor([sos_token].long().to(device)) # (1)

        # Number of sentence to generate
        end_nodes = []

        # Starting node
        node = BeamSearchNode(h=decoder_hidden, prev_node=None, wid=decoder_input, logp=0, length=1)

        # Whole beam search node graph
        nodes = []

        # start the queue
        heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps = 0
        """
        heappush documentation: https://docs.python.org/3/library/heapq.html#heapq.heappush
        heapq.heappush(heap, item)
        Push the value item onto the heap, maintaining the heap invariant.
        """
        # Start beam search
        while True:
            # Give up when decoding takes too long
            if n_dec_steps > max_dec_steps:
                break

            # Fetch the best nodee
            score, _, n = heappop(nodes) # _ is id(node) which is not used
            """
            heapq.heappop(heap) Pop and return the smallest item from the heap, maintaining the heap invariant.
            """
            decoder_input = n.wid # (1)
            decoder_hidden = n.h # (H)

            if n.wid.item() == eos_token and n.prev_node is not None:
                end_nodes.append((score, id(n), n))
                # If we reached maximum # of sentences required
                if len(end_nodes) >= n_best:
                    break
                else:
                    continue

            # Decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden.unsqueeze(0), enc_out) # unsqueeze(0) for batch dim
            """
            decoder_output: (1, V)
            decoder_hidden: (1, H)
            decoder_input: (1)
            decoder_hidden: (1, H)
            enc_out: (T, 1, 2*H)
            """

            # Get topk log probabilities and indices
            topk_log_prob, topk_indecies = torch.topk(decoder_output, beam_width) # (1, beam_width)

            # Then, register new top-k nodes
            for new_k in range(beam_width):
                decoded_t = topk_indecies[0][new_k].view(1) # (1)
                """
                topk_indecies[0][new_k].view(1) means:
                topk_indecies: (1, beam_width)
                topk_indecies[0]: (beam_width)
                topk_indecies[0][new_k]: (1)
                topk_indecies[0][new_k].view(1): (1)
                decoded_t: (1)

                Why should we use view(1)?
                Because we need to use torch.cat((decoder_input, decoded_t)) in the next line.
                """
                logp = topk_log_prob[0][new_k].item()

                node = BeamSearchNode(h=decoder_hidden.squeeze(0),
                                      prev_node=n,
                                      wid=decoded_t,
                                      logp=n.logp+logp,
                                      length=n.length+1)
                heappush(nodes, (-node.eval(), id(node), node))

            # Increase decoding step
            n_dec_steps += 1

        # If no sentence ends within max_dec_steps, force stop at EOS
        if len(end_nodes) == 0:
            end_nodes = [heappop(nodes) for _ in range(beam_width)]

        # Construct sequences from end_nodes
        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes, key=lambda x: x[0]):
            sequence = [n.wid.item()]
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1] # reverse
            """
            reverse example:
            a = [1, 2, 3, 4, 5]
            a[::-1]
            [5, 4, 3, 2, 1]
            """

            n_best_seq_list.append(sequence)

        n_best_list.append(n_best_seq_list)

    return n_best_list

def batch_beam_search_decoding(
        decoder,
        enc_outs,
        enc_last_h,
        beam_width,
        n_best,
        sos_token,
        eos_token,
        max_dec_steps,
        device,
        ):
    
    assert beam_width >= n_best # the purpose is to make sure that n_best is not greater than beam_width

    n_best_list = []
    bs = enc_last_h.shape[0] # batch size

    # Get last encoder hidden state
    decoder_hidden = enc_last_h # (B, H)

    # Prepare first token for decoder
    decoder_input = torch.tensor([sos_token]).repeat(1, bs).long().to(device) # (1, B)
    """Why repeat?
    Because we need to feed the same token to all the batch.
    """

    # Number of sentence to generate
    end_nodes_list = [[] for _ in range(bs)]

    # Whole beam search node graph
    nodes = [[] for _ in range(bs)]

    # Starting the queue
    for bid in range(bs):
        node = BeamSearchNode(h=decoder_hidden[bid], prev_node=None, wid=decoder_input[:, bid], logp=0, length=1)
        heappush(nodes[bid], (-node.eval(), id(node), node))
    
    # Start beam search
    fin_nodes = set() # finished nodes, set() is faster than list()
    history = [None for _ in range(bs)] # history size = batch size
    n_dec_steps_list = [0 for _ in range(bs)]
    """0 for _ in range(bs) means: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Why do we need this?
    Because we need to check whether the number of decoding steps is greater than max_dec_steps or not.
    """
    while len(fin_nodes) < bs:
        decoder_input, decoder_hidden = [], [] # size = batch size
        for bid in range(bs):
            if bid not in fin_nodes and n_dec_steps_list[bid] > max_dec_steps:
                fin_nodes.add(bid)

            if bid in fin_nodes:
                score, n = history[bid] # dummy for data consistency
                """while history has the size is batch size, why can you declare score and n like this?
                Because we will not use score and n in the next line.
                What is the purpose of this line?
                To make the size of decoder_input and decoder_hidden is batch size.
                """
            else:
                score, _, n = heappop[bid]
                if n.wid.item() == eos_token and n.prev_node is not None:
                    end_nodes_list[bid].append((score, id(n), n))
                    if len(end_nodes_list[bid]) >= n_best:
                        fin_nodes.add(bid)
                history[bid] = (score, n)
            decoder_input.append(n.wid)
            decoder_hidden.append(n.h)

        decoder_input = torch.cat(decoder_input).to(device) # (bs)
        decoder_hidden = torch.stack(decoder_hidden, 0).to(device) # (bs, H)
        """stack(decoder_hidden, 0) means: stack along batch dimension, 0 means batch dimension"""

        # Decode for one step using decoder
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, enc_outs) # (bs, V) and (bs, H)

        # Get topk log probabilities and indices
        topk_log_prob, topk_indecies = torch.topk(decoder_output, beam_width) # (bs, beam_width)

        # Then, register new top-k nodes
        for bid in range(bs):
            if bid in fin_nodes:
                continue
            score, n = history[bid]
            if n.wid.item() == eos_token and n.prev_node is not None:
                continue
            for new_k in range(beam_width):
                decoded_t = topk_indecies[bid][new_k].view(1) # (1)
                """why view(1)?
                Because we need to use torch.cat((decoder_input, decoded_t)) in the next line.
                """
                logp = topk_log_prob[bid][new_k].item() 
                """Why should we use .item()?
                Because we need to use logp in the next line.
                """
                node = BeamSearchNode(h=decoder_hidden[bid],
                                      prev_node=n,
                                      wid=decoded_t,
                                      logp=n.logp+logp,
                                      length=n.length+1)
                heappush(nodes[bid], (-node(eval(), id(node), node)))
            n_dec_steps_list[bid] += beam_width

    # Construct sequences from end_nodes
    for bid in range(bs):
        if len(end_nodes_list[bid]) == 0:
            end_nodes_list[bid] = [heappop(nodes[bid]) for _ in range(beam_width)] # size = beam_width
        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes_list[bid], key=lambda x: x[0]): # end_nodes_list size = (bs, n_best)  
            sequence = [n.wid.item()]
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1]

            n_best_seq_list.append(sequence)

        n_best_list.append(copy.copy(n_best_seq_list))

    return n_best_list  


if __name__ == '__main__':
    pass