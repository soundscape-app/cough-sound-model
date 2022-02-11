from cv2 import split
from flask import Flask, request, make_response
from pydub import AudioSegment
import numpy as np
import pyloudnorm as pyln
import math
from typing import *
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from matplotlib import cm
import io
import librosa
import librosa.display
import cv2
from keras.models import load_model
from PIL import Image, ImageOps

app = Flask('caugh')

def split_audio(audio: AudioSegment) -> List[np.ndarray]:
    result = []
    total_mins = math.ceil(audio.duration_seconds // 0.030)
    min_per_split = 1
    for i in range(0, total_mins, min_per_split):
        t1 = i * 30
        t2 = (i+min_per_split) * 30
        split_audio = audio[t1:t2]
        result.append(np.array(split_audio.get_array_of_samples()).astype(np.float32))
    return result

def get_loudness(audios: List[np.ndarray], rate: int):
    result = []
    for audio in audios:
        meter = pyln.Meter(rate, block_size=0.001)
        l = meter.integrated_loudness(audio)
        if l > -100:
            result.append(l)
        else:
            result.append(0)
    scale = min(result)
    for i in range(len(result)):
        if result[i] != 0:
            result[i] -= scale
    return result

def make_image(sq) -> Image:
    x = np.array(sq).T
    X = x[:43]
    X = X.reshape(1, -1)
    image_size = 15
    gasf = GramianAngularField(image_size)
    X_gasf = gasf.fit_transform(X)

    plt.figure(figsize=(16, 16))
    plt.imshow(X_gasf[0], cmap='rainbow',origin='lower')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0, transparent = True, format='png')
    plt.close()
    
    return Image.open(buf)

def make_mel_spectogram(wav: List[int], sr: int) -> Image:
    melspec = librosa.feature.melspectrogram(y=wav, sr=sr)
    logmelspec = librosa.power_to_db(melspec)
    plt.figure(figsize = (30,30))
    plt.axis('off')
    librosa.display.specshow(logmelspec, sr=sr,  x_axis='time',y_axis='mel', cmap=cm.jet)
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0, transparent=True, format='png')
    plt.close()

    return Image.open(buf)

def run_model(cv2_image):
    color_coverted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image=Image.fromarray(color_coverted)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    classes = np.argmax(prediction, axis=1)
    return str(classes[0])
 
@app.route('/')
def health_check():
    return make_response('healthy', 200)

model = load_model('keras_model.h5')

@app.route('/process', methods=['POST'])
def process_audio_file():
    audio = request.files.get('audio')
    audio.save('test.wav')
    
    audio = AudioSegment.from_wav(audio)
    framerate = audio.frame_rate
    audios = split_audio(audio)

    loudness = get_loudness(audios, framerate)
    loudness_image = make_image(loudness)

    wav, sr = librosa.load('test.wav')
    melspec = make_mel_spectogram(wav, sr)
    
    loudness_image = loudness_image.resize((280, 280))
    melspec = melspec.resize((280, 280))

    loudness_image = np.array(loudness_image)
    melspec = np.array(melspec)

    loudness_image = cv2.cvtColor(loudness_image, cv2.COLOR_RGB2BGR)
    melspec = cv2.cvtColor(melspec, cv2.COLOR_RGB2BGR)

    mixed = cv2.addWeighted(melspec, 1, loudness_image, 0.8, 0)
    
    result = {}
    if request.args.get('predict') is not None:
        result = run_model(mixed)

    return make_response(result, 200)
    



if __name__ == '__main__':
    app.run(debug=True)
    # audio = AudioSegment.from_wav('test.wav')
    # rate = audio.frame_rate
    # result = split_audio(audio)
    # loudness = get_loudness(result, rate)
    # loudness_image = make_image(loudness)

    # wav, sr = librosa.load('test.wav')
    # melspec = make_mel_spectogram(wav, sr)
    
    # loudness_image = loudness_image.resize((280, 280))
    # melspec = melspec.resize((280, 280))

    # loudness_image.show('loudness')
    # melspec.show('melspec')

    # loudness_image = np.array(loudness_image)
    # melspec = np.array(melspec)

    # loudness_image = cv2.cvtColor(loudness_image, cv2.COLOR_RGB2BGR)
    # melspec = cv2.cvtColor(melspec, cv2.COLOR_RGB2BGR)

    # mixed = cv2.addWeighted(melspec, 1, loudness_image, 0.8, 0)
    # print(mixed)
    # cv2.imshow('mixed', mixed)
    # cv2.waitKey()
