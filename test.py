import streamlit as st
import time
import numpy as np
import IPython.display as ipd
#ipd.Audio(audio, rate=16000)

from online_scd.model import SCDModel
from online_scd.streaming import StreamingDecoder
import timeit


from online_scd.utils import load_wav_file



def stream(file_name):
    audio_full = open(file_name, 'rb')
    audio_bytes = audio_full.read()
    # audio = np.array(audio_bytes)
    audio_full = load_wav_file(file_name, 16000)
    audio = audio_full
    #audio = audio_full[int(5.88*16000) : 126*16000]
    st.audio(audio_bytes, format='audio/wav')
    # extract around 2 minutes of audio, and remove the start which contains a jingle
    #audio = audio_full[int(5.88*16000) : 126*16000]
    
    model = SCDModel.load_from_checkpoint("test/sample_model/checkpoints/epoch=102.ckpt")
    streaming_decoder = StreamingDecoder(model)
    speaker_change_points = []

    #status_text = st.sidebar.empty()
    last_rows = np.zeros((1,1))
    chart = st.line_chart(last_rows)

    # for p in speaker_change_points:
    #     samples = int(p * 16000)
    #     audio[samples - 100: samples+100] = np.random.random(200)        
    #audio = audio_full[int(5.88*16000) : 126*16000]
    streaming_decoder = StreamingDecoder(model)
    frame_number = 0
    # stream in 1000 sample chunks, varying size chunks are accepted
    for i in range(0, len(audio), 1000):
        start = timeit.timeit()
        for probs in streaming_decoder.process_audio(audio[i: i+1000]):
            #   if probs[1] > 0.5:
                #     print(f"Speaker break with probatimbility {probs[1]} at frame {frame_number} (1 frame = 100 ms)")
            new_rows = np.zeros((1, 1))
            new_rows[0,0] = probs[1].detach().numpy()
            #new_rows = np.expand_dims(new_rows, axis=(0, 1))
            #breakpoint()
            chart.add_rows(new_rows)

            
            frame_number += 1
        end = timeit.timeit()
        time.sleep(1/16-end+start)
        # n is not connected to any other logic, it just causes a plain
# rerun.
    st.button("Re-run")


def main():
    option = st.sidebar.selectbox(
        'Which audio file would you like to use?',
        ('paevakaja', 'osoon'), 0)
    if option == 'paevakaja':
        file_name = "test/sample_dataset/71_ID117_344945.wav"
    elif option == 'osoon':
        file_name = "test/sample_dataset/3321821.wav"
    stream(file_name)

if __name__ == "__main__":
    main()