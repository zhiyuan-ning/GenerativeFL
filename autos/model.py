import torch
import torch
import torch.nn as nn

from autos.decoder import construct_decoder
from autos.encoder import construct_encoder
SOS_ID = 0
EOS_ID = 0


# gradient based automatic device selection
class AUTOS(nn.Module):
    def __init__(self,Config):
        super(AUTOS, self).__init__()
        self.style = Config().server.autos.method_name
        self.gpu = Config().server.autos.gpu
        self.encoder = construct_encoder(Config)
        self.decoder = construct_decoder(Config)
        if self.style == 'rnn':
            self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, feat_emb, predict_value = self.encoder.forward(input_variable)
        decoder_hidden = (feat_emb.unsqueeze(0), feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        feat = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, feat

    def generate_new_device(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, feat_emb, predict_value, new_encoder_outputs, new_feat_emb = \
            self.encoder.infer(input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_feat_emb.unsqueeze(0), new_feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(None, new_encoder_hidden, new_encoder_outputs)
        new_feat_seq = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_feat_seq

