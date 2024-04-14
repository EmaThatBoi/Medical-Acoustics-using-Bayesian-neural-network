import numpy as np
import pandas as pd
import os
from librosa import load, feature, power_to_db
import functions
from sklearn.preprocessing import OneHotEncoder

# Caricamento dei dati dei pazienti
patient_diagnosis = pd.read_csv('ICBHI set/ICBHI_patient_diagnosis.csv', sep='\t', names=('ID', 'Diagnosis'))
patient_diagnosis = np.array(patient_diagnosis)

# Caricamento dei pazienti per train set e test set
set_differences = pd.read_csv('ICBHI set/ICBHI_filename_differences.csv', sep='\t', names=('name', 'set'))
set_differences = set_differences.sort_values(['name'])
set_differences = np.array(set_differences)

diagnosis = np.array(['Healthy', 'Diseased'])
results = []
index = -1
conta = 0
persi = []

for file_name in os.listdir('ICBHI set/ICBHI_final_database'):

    if file_name.endswith('.txt'):
        index = index + 1
        print(f'Processing file: {file_name}')  # stampa il nome del file in lavorazione
        breathing = pd.read_csv(('ICBHI set/ICBHI_final_database/'+file_name),sep='\t', names=('start','end','s','c'))
        start = [breathing['start'][i] for i in range(len(breathing))]
        end = [breathing['end'][i] for i in range(len(breathing))]

        for id, diagnosi in patient_diagnosis:
            if str(file_name)[0:3] == str(id):  # se è lo stesso paziente 101 == 101
                print(f'Matched patient ID: {id} with file: {file_name}')  # stampa ID paziente corrispondente

                path = os.path.join('ICBHI set/ICBHI_final_database/', str(file_name[0:-4] + '.wav'))
                samples, sr = load(path, sr=44100)
                for i in range(len(start)):
                    start_sample = int(start[i] * sr)
                    end_sample = int(end[i] * sr)
                    breath_cycle_samples = samples[start_sample:end_sample]
                    signal = feature.melspectrogram(y=breath_cycle_samples, sr=44100, hop_length=256)
                    signal_dB = power_to_db(signal)
                    if signal_dB.shape[1] < 500:  # se array è più corto degli altri paddalo
                        signal_dB = np.pad(signal_dB, ( (0,0), (0, 500 - signal_dB.shape[1]) ),mode='constant')
                    np.matrix.resize(signal_dB,(128,500))
                    results.append(signal_dB)

                    if file_name[:-4] == set_differences[index][0]:
                        set_type= set_differences[index][1]
                        if diagnosi == 'Healthy':
                            directory = diagnosis[0]  # Healthy
                            if not os.path.exists(os.path.join('official set/',set_type ,directory)):  # crea directory se non esiste
                                print(f'Creating directory: official set/{set_type}/{directory}')  # stampa la directory in creazione
                                os.makedirs('official set/'+set_type+'/' +directory)
                        else:
                            directory = diagnosis[1]  # Diseased
                            if not os.path.exists(os.path.join('official set/',set_type, directory)):  # crea directory se non esiste
                                print(f'Creating directory: official set/{set_type}/{directory}')  # stampa la directory in creazione
                                os.makedirs('official set/'+set_type+'/' +directory)

                        np.save(os.path.join('official set/',set_type ,directory, file_name[:-4] +'part_'+str(i) +'.npy'), signal_dB)
                    else:
                        conta = conta + 1
                        persi.append((file_name[:-4],set_differences[index][0]))
#######################################
train_dir_path = 'official set/train'
train_dir_label = os.listdir('official set/train')

x_train_data,y_train_data = functions.training_set(train_dir_path,train_dir_label)

test_dir_path = 'official set/test'
test_dir_label = os.listdir('official set/test')

x_test_data,y_test_data = functions.test_set(test_dir_path,test_dir_label)