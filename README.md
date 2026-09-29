# 🚀 Industrial Defect Classifier: Dai Dati al Deep Learning

Benvenuti nel repository del progetto **Industrial Defect Classifier**. Questo progetto documenta lo sviluppo di un sistema di computer vision e intelligenza artificiale per il controllo qualità industriale, strutturato attraverso fasi progressive: dalla preparazione delle immagini alla comparazione tra algoritmi classici di Machine Learning e reti neurali di Deep Learning.

---

## 🛠️ Organizzazione dei Moduli

### 🔹 Fase 1: Ingestion e Preprocessing Dati
* **Modulo 01: Validazione Percorsi** - Controllo di sicurezza e integrità delle directory locali delle immagini.
* **Modulo 02: Image Processing con Pillow** - Lettura, ridimensionamento e conversione dei file grafici in matrici NumPy.
* **Modulo 03: Normalizzazione Numerica** - Riscalatura dei pixel da [0–255] all'intervallo `[0.0, 1.0]` e flattening a 4096 dimensioni.
* **Modulo 04: Serializzazione Compressa** - Creazione automatica dell'archivio binario compatto `factory_dataset.npz` ad accesso rapido.

### 🔹 Fase 2: Architettura e Standard di Sviluppo
* **Modulo 05: Tipizzazione Forte (Type Hints)** - Utilizzo rigoroso di annotazioni di tipo (`np.ndarray`, `Image.Image`) per un codice robusto.
* **Modulo 06: Logging Strutturato** - Tracciamento degli stati di esecuzione su console e su file tramite modulo `logging`.
* **Modulo 07: Separazione delle Pipeline** - Disaccoppiamento tra preprocessing dei dati e sessione di addestramento per ottimizzare la RAM.
* **Modulo 08: Split Stratificato** - Bilanciamento rigoroso tra classi (Sani vs Difettosi) per i set di train e test.

### 🔹 Fase 3: Machine Learning & Spiegabilità (XAI)
* **Modulo 09: Decision Tree Classifier** - Addestramento dell'albero decisionale basato su impurità di Gini e profondità controllata.
* **Modulo 10: Feature Importance Heatmap** - Estrazione visiva con Seaborn dei pixel discriminanti che causano l'anomalia.
* **Modulo 11: Metriche di Performance** - Valutazione comparativa su accuratezza, precisione e matrice di confusione.

### 🔹 Fase 4: Deep Learning con Reti Neurali
* **Modulo 12: Architettura Keras MLP** - Progettazione di un percettrone multistrato (4096 nodi input, Dense 32 nodi con ReLU).
* **Modulo 13: Ottimizzazione e Convergenza** - Configurazione di Adam (`lr=0.0001`) e funzione di costo Binary Cross-Entropy.
* **Modulo 14: Validazione e Confronto** - Analisi comparativa diretta delle performance tra albero decisionale e rete neurale.

---

## 🏆 Progetto Finale: Pipeline Completa di Classificazione
Il flusso operativo culmina nell'esecuzione integrata dei due script applicativi:

1. **Preprocessing Pipeline (`Reti_neurali_preprocessing.py`)**: Elaborazione batch, normalizzazione ed export dei vettori numerici.
2. **Model Training & Benchmark (`Reti_neurali_testing.py`)**: Addestramento simultaneo di Keras MLP e Decision Tree con generazione della mappa di calore visiva.

---

## 🎓 Competenze Acquisite
* **Linguaggi:** Python 3.x (Modular Design, Strong Typing, Production Logging).
* **AI & Machine Learning:** TensorFlow / Keras, Scikit-Learn.
* **Computer Vision & Dati:** NumPy, Pillow (PIL).
* **Visualizzazione:** Matplotlib, Seaborn.

---

**Progetto realizzato da Mattia Dellanoce**
