import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import sys
import re
import json
import math
import torch
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
    text = re.sub('[\s+]', ' ', text)
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# Inference
hps = utils.get_hparams_from_file("./configs/yuuka.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.nspeakers,
    **hps.model)
 = netg.eval()

 = utils.load_checkpoint(f"/content/drive/MyDrive/drive/MyDrive/politic/G{sys.argv[1]}.pth", net_g, None)

text = '''
[JA]しばらく焚火の跡をかき回してみたが、残念ながら灰の中には食べられそうなものは残っていなかった。
'''

speed = 1
for idx in range(12):
    sid = torch.LongTensor([idx])
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1/speed)[0][0,0].data.cpu().float().numpy()
    write(f'/content/drive/MyDrive/vitsoutput2/output{idx}.wav', hps.data.samplingrate, audio)
    print(f'output{idx} 음성 합성 완료')
    break