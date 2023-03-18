import numpy as np
import pandas as pd
import os
import librosa
from librosa import display
import matplotlib.pyplot as plt

def spectrogram(path):
    samples, sr = librosa.load(path, sr=None)
    signal = librosa.feature.melspectrogram(y=samples,
                                            sr=sr)
    signal_dB = librosa.power_to_db(signal)

    #fig = plt.subplot()
    #spectrogram = librosa.display.specshow(signal_dB)

    return signal_dB


if __name__ == '__main__':
    patient_diagnosis = pd.read_csv('archive/patient_diagnosis.csv', names=('ID', 'Diagnosis'))
    patient_diagnosis = np.array(patient_diagnosis)

    diagnosis = []
    for el in patient_diagnosis[:, 1]:
        if el in diagnosis:
            pass
        else:
            diagnosis.append(el)

    diagnosis = np.array(diagnosis)

    results = []
    for file_name in os.listdir('archive/audio_and_txt_files/'):
        if file_name.endswith('.wav'):
            for nome_diagnosi in patient_diagnosis:
                if str(file_name)[0:3] == str(nome_diagnosi[0]):                      # se è lo stesso paziente 101 == 101
                    directory = str(nome_diagnosi[1])                               # assegna nome directory
                    if not os.path.exists(os.path.join('diagnosi/', directory)):    # crea directory se non esiste
                        os.mkdir('diagnosi/'+directory)                             # crea solo directory poichè mkdir crea il path più a destra

                    result = spectrogram(os.path.join('archive/audio_and_txt_files/', file_name))
                    if result.shape[1] < 1723:                                      # se array è più corto degli altri paddalo
                        result = np.pad(result, ( (0,0), (0, 1723 - result.shape[1]) ),mode='constant')

                    np.save(os.path.join('diagnosi/', directory, file_name[:-4] + '.npy'), result)
                    results.append(result)








