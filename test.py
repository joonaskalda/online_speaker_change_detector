import streamlit as st
import time
import numpy as np

from online_scd.utils import load_wav_file
audio_full = open("test/sample_dataset/71_ID117_344945.wav", 'rb')
audio_bytes = audio_full.read()
audio = np.array(audio_bytes)
audio_full = load_wav_file("test/sample_dataset/71_ID117_344945.wav", 16000)

audio = audio_full[int(5.88*16000) : 126*16000]
st.audio(audio_bytes, format='audio/wav')
# extract around 2 minutes of audio, and remove the start which contains a jingle
#audio = audio_full[int(5.88*16000) : 126*16000]
import IPython.display as ipd
#ipd.Audio(audio, rate=16000)

from online_scd.model import SCDModel
from online_scd.streaming import StreamingDecoder

model = SCDModel.load_from_checkpoint("test/sample_model/checkpoints/epoch=102.ckpt")
streaming_decoder = StreamingDecoder(model)
speaker_change_points = []

status_text = st.sidebar.empty()
last_rows = np.zeros((1,1))
chart = st.line_chart(last_rows)

# for p in speaker_change_points:
#     samples = int(p * 16000)
#     audio[samples - 100: samples+100] = np.random.random(200)        
audio = audio_full[int(5.88*16000) : 126*16000]
streaming_decoder = StreamingDecoder(model)
frame_number = 0
# stream in 1000 sample chunks, varying size chunks are accepted
for i in range(0, len(audio), 1000):
    for probs in streaming_decoder.process_audio(audio[i: i+1000]):
    #   if probs[1] > 0.5:
    #     print(f"Speaker break with probability {probs[1]} at frame {frame_number} (1 frame = 100 ms)")
      new_rows = np.zeros((1, 1))
      new_rows[0,0] = probs[1].detach().numpy()
      #new_rows = np.expand_dims(new_rows, axis=(0, 1))
      #breakpoint()
      chart.add_rows(new_rows)

      
      frame_number += 1
      time.sleep(0.065)
# n is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")