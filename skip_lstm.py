## use pytorch lstm as the lstm
## sees slower than use torch LSTMCell to build lstm
## use c1[-1] c0[-1] to compute the probability. use c0[-1] to compute the threshold


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class RoundFunctionST(Function):

    @staticmethod
    def forward(ctx, input, threshold):
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

RoundST = RoundFunctionST.apply

class skip_torchlstm_cell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first = False, dropout=0, bidirectional=False):
        super(skip_torchlstm_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.torchLSTM = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

        self.skip_gate_inner_linear_prev = nn.Linear(hidden_size , hidden_size  /2)
        self.skip_gate_inner_linear_curr = nn.Linear(hidden_size , hidden_size /2)
        self.skip_gate_LReLU = nn.LeakyReLU()
        self.skip_gate_out_linear = nn.Linear(hidden_size /2, 1)
        self.skip_gate_Sigmoid= nn.Sigmoid()


        # Although the gradient for the threshold is None,
        # the input of this MLP keeps update and leads to better results than a fixed threshold....
        # however, still need to find a better way of setting the threshold...
        # using the fixed threshold 0.5 is much better than the Vanilla LSTM, but no better than this MLP...
        self.threshold_gate_inner_linear = nn.Linear(hidden_size , hidden_size /2)
        self.threshold_gate_LReLU = nn.LeakyReLU()
        self.threshold_gate_out_linear = nn.Linear(hidden_size /2, 1)
        self.threshold_gate_Sigmoid= nn.Sigmoid()

        self.bn = RoundST

    def forward(self, input, hidden):
        #INPUT
        # input: batch * input_size
        # hidden: h_0, num_layers * batch * hidden_size
        #         c_0, num_layers * batch * hidden_size
        #         current_update_prob, batch * 1
        #OUTPUT
        # new_h: num_layers * batch_size * hidden_size
        # new_c: num_layers * batch_size * hidden_size
        # next_update_prob: batch_size * 1
        h_0, c_0, current_update_cum = hidden
        batch_size, hidden_size = input.shape
        input = input.view(1, batch_size, hidden_size)
        output, (h_1, c_1) = self.torchLSTM(input, (h_0, c_0))
        # output: 1 * batch * hidden_size
        # h: num_layers * batch * hidden_size; h[-1] = output.view(batch_size, hidden_size)
        # c: num_layers * batch * hidden_size

        # compare c0 with c1 to get the update_prob_delta
        # compare c0 with c1 to get the update_prob_delta


        skip_gate_h = self.skip_gate_inner_linear_prev(c_0[-1]) + self.skip_gate_inner_linear_curr(c_1[-1])
        skip_gate_h = self.skip_gate_LReLU(skip_gate_h)
        skip_gate_h2 = self.skip_gate_out_linear(skip_gate_h)
        update_prob_delta = self.skip_gate_Sigmoid(skip_gate_h2)

        update_threshold_h = self.threshold_gate_inner_linear(c_0[-1])
        update_threshold_h = self.threshold_gate_LReLU(update_threshold_h)
        update_threshold_h2 = self.threshold_gate_out_linear(update_threshold_h)
        update_threshold = self.threshold_gate_Sigmoid(update_threshold_h2)

        update_prob = (current_update_cum + torch.min(update_prob_delta, 1.0-current_update_cum))


        update_gate = self.bn(update_prob,update_threshold)

        # apply the update_gate
        new_h = update_gate * h_1 + (1.0- update_gate) * h_0
        new_c = update_gate * c_1 + (1.0- update_gate) * c_0

        # calculate the update_cum; if updated, set to 0
        update_cum = 0.0 * update_gate + (1.0 - update_gate) * update_prob

        return new_h, new_c, update_cum, update_prob_delta, update_prob

        def update_gate_weight(self):
            print(self.skip_gate_linear.weight)

def test_skip_torchlstm_cell(seq_len, batch_size, input_size, hidden_size, num_layers):
    input = torch.randn([seq_len, batch_size, input_size])
    hx = torch.zeros([num_layers, batch_size, hidden_size])
    cx = torch.zeros([num_layers, batch_size, hidden_size])
    #update_prob = torch.ones([batch_size, hidden_size])
    update_prob = torch.tensor([[1.0],[0.5], [0.0]]) # here batch=3
    cell = skip_torchlstm_cell(input_size, hidden_size, num_layers)
    for i in range(seq_len):
        hx, cx, update_prob = cell(input[i], (hx, cx, update_prob))
        print("----hidden----")
        print(hx)
        print(hx.shape)
        print("----hidden----")
        print("----update prob----")
        print(update_prob)
        print(update_prob.shape)
        print("----update prob----")
    loss = hx[-1].mean()
    loss.backward()

#test_skip_torchlstm_cell(seq_len=2, batch_size=3, input_size=2, hidden_size=5, num_layers=2)

class skip_torchlstm_cell_(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first = False, dropout=0, bidirectional=False):
        super(skip_torchlstm_cell_, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.skip_lstm = skip_torchlstm_cell(input_size, hidden_size, num_layers, bias, False, dropout, bidirectional=False)

    def update_gate_weight(self):
        print(self.skip_lstm.update_gate_weight())

    def forward(self, input, hidden=None):
        #INPUT
        # input: seq* batch * input_size or seq* batch * input_size
        # hidden: h_0, num_layers * batch * hidden_size
        #         c_0, num_layers * batch * hidden_size
        #         current_update_prob, batch * 1
        #OUTPUT
        # output: seq* batch * input_size or seq* batch * input_size
        # new_h: num_layers * batch_size * hidden_size
        # new_c: num_layers * batch_size * hidden_size
        # next_update_prob: batch_size * 1

        if self.batch_first:
            batch_size, seq_len, input_size = input.shape
        else:
            seq_len, batch_size, input_size = input.shape

        if hidden is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
            prob = input.new_ones(batch_size, 1)
            h_in = (hx, cx, prob)
        else:
            h_in = hidden
        output_list = []
        update_delta_list = []
        update_prob_list = []
        update_cum_list = []
        is_print = not self.training
        for t in range(seq_len):
            if self.batch_first:
                current_input = input[:,t,:]
            else:
                current_input = input[t,:,:]
            hx, cx, update_cum, update_delta, update_prob_before_bn = self.skip_lstm(current_input, h_in)
            if is_print:
                update_delta_list.append(update_delta.tolist()[0][0])
                update_prob_list.append(update_prob_before_bn.tolist()[0][0])
                update_cum_list.append(update_cum.tolist()[0][0])
            h_in = (hx, cx, update_cum)
            output_list.append(hx[-1]) #hx[-1] batch * hidden
        output = torch.stack(output_list, dim=0)
        if self.batch_first:
            output = output.transpose(0,1)

        if is_print:
            print("-------update_prob_delta-------")
            print(update_delta_list)
            print("-------update_prob-------")
            print(update_prob_list)
            print("-------update_cum_list-------")
            print(update_cum_list)
        return output, (hx, cx)

def test_skip_torchlstm_cell_(seq_len, batch_size, input_size, hidden_size, num_layers):
    input = torch.randn([seq_len, batch_size, input_size])
    hx = torch.zeros([num_layers, batch_size, hidden_size])
    cx = torch.zeros([num_layers, batch_size, hidden_size])
    #update_prob = torch.ones([batch_size, hidden_size])
    update_prob = torch.tensor([[1.0],[0.5], [0.0]]) # here batch=3
    cell = skip_torchlstm_cell_(input_size, hidden_size, num_layers=num_layers)
    output, (hn, cn) = cell(input, (hx, cx, update_prob))
    print("-----output-----")
    print(output)
    print(output.shape)
    print("-----hn-----")
    print(hn)
    print(hn.shape)
    loss = output[-1].mean()
    loss.backward()

#test_skip_torchlstm_cell_(seq_len=4, batch_size=3, input_size=2, hidden_size=5, num_layers=2)

class skip_lstm_ESPnet(torch.nn.Module):
    # make is suitable for ESPnet
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(skip_lstm_ESPnet, self).__init__()
        self.cdim = cdim
        self.skip_lstm = skip_torchlstm_(idim, cdim, elayers, batch_first=True,
                                    dropout=dropout, bidirectional=False)
        self.l_last = torch.nn.Linear(cdim , hdim)

    def forward(self, xpad, ilens):
        '''BLSTM forward

        :param xs:
        :param ilens:
        :return:
        '''
        # if the number of non padding elements in the longest seq in xpad is not equal to seq_len
        # pack_padded_sequence will have different seq size to make sure there is no padding elements
        # in the longest sequence... so this implementation may be dangerous
        # mask_pack = pack_padded_sequence(mask, ilens, batch_first=True)

        # to avoid this, check if the first element in ilens is equal to seq_len
        # if not, narrow the input to ilen

        batch_size, seq_len, input_size = xpad.shape

        # narrow the input if seq_len not equal to ilens[0]
        if not ilens[0] == seq_len:
            xpad = torch.narrow(xpad, 1, 0, ilens[0])
            batch_size, seq_len, input_size = xpad.shape

        mask = xpad.new_ones(batch_size, seq_len, self.cdim)

        mask_pack = pack_padded_sequence(mask, ilens, batch_first=True)
        mask, mlens = pad_packed_sequence(mask_pack, batch_first=True)

        ys, (hy, cy) = self.skip_lstm(xpad)

        del hy, cy
        ypad = mask * ys

        projected = torch.tanh(self.l_last(
            ypad.contiguous().view(-1, ypad.size(2))))
        xpad = projected.view(ypad.size(0), ypad.size(1), -1)
        #return ypad, ilens  # for testing. check if mask for padding works.
        return xpad, ilens  # xpad: b x sequence_length x dim; ilens: tensor([len_1, len_2,...,len_n])

def test_skip_lstm_ESPnet(batch_first, seq_len, batch_size, input_size, hidden_size, num_layers, dropout):
    if batch_first:
        input = torch.randn([batch_size, seq_len, input_size])
    else:
        input = torch.randn([seq_len, batch_size, input_size])
    hx = torch.zeros([num_layers, batch_size, hidden_size])
    cx = torch.zeros([num_layers, batch_size, hidden_size])
    #update_prob = torch.ones([batch_size, hidden_size])
    update_prob = torch.tensor([[1.0],[0.5], [0.0]]) # here batch=3
    net_work = skip_lstm_ESPnet(input_size, num_layers, hidden_size, hidden_size, dropout)
    xpad, ilens = net_work(input, torch.tensor([3,2,1]))
    print("-----input-----")
    print(input)
    print(input.shape)
    print("-----xpad-----")
    print(xpad)
    print(xpad.shape)
    loss = xpad[-1].mean()
    loss.backward()

#test_skip_lstm_ESPnet(batch_first=True, seq_len=5,batch_size=3,input_size=2,hidden_size=4,num_layers=2,dropout=0.2)
