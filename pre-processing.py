import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from librosa import load,feature,power_to_db
import functions

patient_diagnosis = pd.read_csv('archive/patient_diagnosis.csv', names=('ID', 'Diagnosis'))
patient_diagnosis = np.array(patient_diagnosis)

# x multi class
#diagnosis = []
#for el in patient_diagnosis[:, 1]:
#    if el in diagnosis:
#        pass
#    else:
#        diagnosis.append(el)


diagnosis = np.array(['Healthy', 'Diseased'])
results = []
for file_name in os.listdir('archive/audio_and_txt_files/'):

    if file_name.endswith('.txt'):
        breathing = pd.read_csv(('archive/audio_and_txt_files/'+file_name),sep='\t', names=('start','end','s','c'))
        start = [breathing['start'][i] for i in range(len(breathing))]
        end = [breathing['end'][i] for i in range(len(breathing))]

        for id, diagnosi in patient_diagnosis:
            if str(file_name)[0:3] == str(id):                      # se è lo stesso paziente 101 == 101
                #directory = str(nome_diagnosi[1])     x MULTI CLASS                         # assegna nome directory
                #if not os.path.exists(os.path.join('unofficial set/', directory)):    # crea directory se non esiste
                    #os.mkdir('unofficial set/'+directory)                             # crea solo directory poichè mkdir crea il path più a destra

                path = os.path.join('archive/audio_and_txt_files/', str(file_name[0:-4] + '.wav'))
                samples, sr = load(path, sr=44100)
                for i in range(len(start)):
                    start_sample = int(start[i] * sr)
                    end_sample = int(end[i] * sr)
                    breath_cycle_samples = samples[start_sample:end_sample]
                    signal = feature.melspectrogram(y=breath_cycle_samples, sr=44100, hop_length=256)
                    signal_dB = power_to_db(signal)
                    if signal_dB.shape[1] < 500:                                      # se array è più corto degli altri paddalo
                        signal_dB = np.pad(signal_dB, ( (0,0), (0, 500 - signal_dB.shape[1]) ),mode='constant')
                    np.matrix.resize(signal_dB,(128,500))
                    results.append(signal_dB)

                    if diagnosi == 'Healthy':
                        directory = diagnosis[0]  # Healthy
                        if not os.path.exists(os.path.join('unofficial set/', directory)):  # crea directory se non esiste
                            os.mkdir('unofficial set/' + directory)
                    else:
                        directory = diagnosis[1]  # Diseased
                        if not os.path.exists(os.path.join('unofficial set/', directory)):  # crea directory se non esiste
                            os.mkdir('unofficial set/' + directory)
                    np.save(os.path.join('unofficial set/' ,directory, file_name[:-4] +'part_'+str(i) +'.npy'), signal_dB)

dir_path = 'unofficial set/'
dir_label = os.listdir('unofficial set/')

# estrae le matrici degli spettrogrammi dalle directory
data = functions.create_data(dir_path, dir_label)

#standardizza il dataset (media vicina a 0 e std a 1 su tutto il dataset )
media = np.array([np.mean(arr) for arr,_ in data])
std = np.array([np.std(arr,ddof=1) for arr, _ in data])

for i in range(len(data)):
    arr, label = data[i]
    matrice_standard = ((arr - media[i]) / std[i])
    data[i] = (matrice_standard, label)

# crea la coppia - (spettrogramma, label corrispondente)

x_data, y_data = functions.prepare_data(data, dir_label)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

np.save(file='unofficial set/dataset',arr=x_data)
np.save(file='unofficial set/labels',arr=y_data)