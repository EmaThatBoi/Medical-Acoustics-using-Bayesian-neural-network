# Medical-Acoustics-using-Bayesian-neural-network

### Database: ICBHI 2017 Challenge: Respiratory Sound Database 
il database presenta le registrazioni di 126 pazienti, 920 registrazioni in totale, ci sono 8 categorie di diagnosi possibili:

<code>'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'LRTI', 'Pneumonia', 'URTI' </code>

il numero di registrazioni per ogni paziente è variabile, per esempio, il paziente 103 ha una sola registrazione, ed è l'unico a soffrire di <code>Asthma</code>. 
64 pazienti su 126 soffrono di <code>COPD</code> come conseguenza del fumo.

### pre-processing.py 
il codice si occupa di estrarre dalle registrazioni i mel_spectrogram in dB e di filtrare i 920 spettrogrammi(matrici numpy) in directory suddivise per diagnosi, quindi 8 directory ognuna contenente gli spettrogrammi in base alla diagnosi. Ciò l'ho usato per creare il dataset con le label in formato one-hot-encoded

### functions.py
all'interno del file ci sono due funzioni, che si occupano di creare la coppia <code><spettrogramma,label_diagnosi></code> che poi diventeranno la coppia x_data,y_data 

### main.py 
contiene il codice del modello, attualmente sto lavorando sullo sbilanciamento del dataset, al fine di ottenere dei risultati corretti. 

il dataset è stato standardizzato. 
