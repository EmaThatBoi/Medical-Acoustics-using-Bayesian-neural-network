import numpy as np
import pandas as pd
import os
import librosa
from librosa import display
import matplotlib.pyplot as plt
from multiprocessing import Pool

def spectrogram(path):
    samples, sr = librosa.load(path, sr=None)
    signal = librosa.feature.melspectrogram(y=samples,
                                            sr=sr)
    signal_dB = librosa.power_to_db(signal)

    fig = plt.subplot()
    spectrogram = librosa.display.specshow(signal_dB)

    return spectrogram


if __name__ == '__main__':
    txt = pd.read_csv('archive/patient_diagnosis.csv', names=('ID', 'Diagnosis'))
    txt = np.array(txt)

    diagnosis = []
    for el in txt[:, 1]:
        if el in diagnosis:
            pass
        else:
            diagnosis.append(el)

    diagnosis = np.array(diagnosis)

    pool = Pool(processes=32)
    results = []

    for spettro in os.listdir('archive/audio_and_txt_files/'):
        if spettro.endswith('.wav'):

            for nome_diagnosi in txt:
                if str(spettro)[0:3] == str(nome_diagnosi[0]):                      # se è lo stesso paziente 101 == 101
                    directory = str(nome_diagnosi[1])                               # assegna nome
                    if not os.path.exists(os.path.join('diagnosi/', directory)):    # crea directory se non esiste
                        os.mkdir('diagnosi/'+directory)                             # crea solo directory poichè mkdir crea il path più a destra
                    result = pool.apply_async(spectrogram(os.path.join('archive/audio_and_txt_files/',spettro))
                                              , args=(spettro,))
                    results.append(result)
                    plt.savefig((os.path.join('diagnosi/', directory, spettro[:-4]+'.png')))
                    plt.close()