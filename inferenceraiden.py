#matplotlib inline
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import json
import math
import torch
import sys
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("./configs/raiden.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()


_ = utils.load_checkpoint(f"/content/drive/MyDrive/drive/MyDrive/raiden/G_{sys.argv[1]}.pth", net_g, None)


stn_tst = get_text("[KO]내가 누군가를 좋아한다는 사실이, 그 사람에게는 상처가 될 수 있잖아요.[KO]", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1.1)[0][0,0].data.cpu().float().numpy()
    write(f'/content/drive/MyDrive/vitsoutput3/output0.wav', hps.data.sampling_rate, audio)

stn_tst = get_text("[KO] 독도는 우리 땅입니다. 그냥 우리 땅이 아니라 40년 통한의 역사가 뚜렷하게 새겨져 있는 역사의 땅입니다. 독도는 일본의 한반도 침탈 과정에서 가장 먼저 병탄되었던 우리 땅입니다. 일본이 러일전쟁 중에 전쟁 수행을 목적으로 편입하고 점령했던 땅입니다. 러일전쟁은 제국주의 일본이 한국에 대한 지배권을 확보하기 위해 일으킨 한반도 침략전쟁입니다.[KO]", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1.1)[0][0,0].data.cpu().float().numpy()
    write(f'/content/drive/MyDrive/vitsoutput3/output1.wav', hps.data.sampling_rate, audio)
