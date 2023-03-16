# Medical-Acoustics-using-Bayesian-neural-network

### Database: Respiratory Sound Database (Kaggle)
il database presenta le registrazioni di 126 pazienti, 920 registrazioni in totale, ci sono 8 categorie di diagnosi possibili:

<code>'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'LRTI', 'Pneumonia', 'URTI' </code>

il numero di registrazioni per ogni paziente è variabile, per esempio, il paziente 103 ha una sola registrazione, ed è l'unico a soffrire di <code>Asthma</code>. 
64 pazienti su 126 soffre di <code>COPD</code> come conseguenza del fumo.

### pre-processing.py 
all'interno del file, il codice si occupa di estrarre dalle registrazioni i mel_spectrogram in dB e di ordinare i 920 spettrogrammi in directory suddivise per diagnosi, quindi 8 directory. 

### functions.py
all'interno del file ci sono due funzioni, che si occupano di creare la coppia <code><spettrogramma,label_diagnosi></code> che poi diventeranno la coppia x_train,y_train 

### main.py 
contiene il codice del modello
