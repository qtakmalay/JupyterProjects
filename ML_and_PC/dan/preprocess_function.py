from data import load_wav_16k_mono
import tensorflow as tf
from data import ferseher_tf
import matplotlib.pyplot as plt
import librosa
from data import data
def preprocess(file_path, label):
    str(file_path).replace("._","")
    wav = tf.py_function(func=load_wav_16k_mono, inp=[file_path], Tout=tf.float32)
    wav = wav[:17600]
    spectogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectogram = tf.abs(spectogram)
    spectogram = tf.expand_dims(spectogram, axis=2)
    return spectogram, label


filepath, label = data.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectogram, label = preprocess(filepath, label)




# plt.figure(figsize=(30, 20))
# plt.imshow(tf.transpose(spectogram)[0])
# plt.show()
