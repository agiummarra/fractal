# 🎨 Generatore di Frattali

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

Un'applicazione web interattiva per generare e visualizzare diversi tipi di frattali matematici. Creata con Streamlit e Python.

![App Screenshot](screenshot.png)

## 🌟 Caratteristiche

- **Interfaccia Intuitiva**: UI semplice e user-friendly con controlli interattivi
- **Molteplici Frattali**: Supporto per diversi tipi di frattali classici:
  - 🌀 Insieme di Mandelbrot
  - 🌊 Insieme di Julia
  - 🔲 Tappeto di Sierpinski
  - ❄️ Fiocco di Koch
  - 🌿 Felce di Barnsley
- **Personalizzazione**: Controllo completo sui parametri di generazione
- **Visualizzazione in Tempo Reale**: Rendering immediato delle modifiche
- **Formule Matematiche**: Visualizzazione delle formule matematiche in LaTeX
- **Mappe Colori**: Diverse opzioni di colorazione per i frattali

## 🚀 Installazione

1. Clona il repository:

```bash
git clone https://github.com/agiummarra/fractal.git
cd fractal
```

2. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

3. Avvia l'applicazione:

```bash
streamlit run app.py
```

## 📦 Requisiti

- Python 3.8 o superiore
- Dipendenze principali:
  - streamlit >= 1.28.0
  - numpy >= 1.24.0
  - matplotlib >= 3.7.0
  - Pillow >= 10.0.0

## 🎮 Utilizzo

1. Seleziona il tipo di frattale dalla barra laterale
2. Modifica i parametri disponibili:
   - Dimensione dell'immagine
   - Numero di iterazioni
   - Limiti del piano complesso (per Mandelbrot e Julia)
   - Mappa colori
3. Clicca su "Genera Frattale" per visualizzare il risultato

## 🔬 Dettagli Matematici

### Insieme di Mandelbrot

L'insieme di Mandelbrot è definito dalla formula:

```math
z_{n+1} = z_n^2 + c
```

dove `c` è un numero complesso e la sequenza inizia con `z₀ = 0`.

### Insieme di Julia

Simile all'insieme di Mandelbrot, ma con un valore `c` fisso:

```math
z_{n+1} = z_n^2 + c
```

### Felce di Barnsley

Utilizza un sistema di funzioni iterate (IFS) con quattro trasformazioni affini:

```math
f_1(x,y) = (0, 0.16y)
f_2(x,y) = (0.85x + 0.04y, -0.04x + 0.85y + 1.6)
f_3(x,y) = (0.2x - 0.26y, 0.23x + 0.22y + 1.6)
f_4(x,y) = (-0.15x + 0.28y, 0.26x + 0.24y + 0.44)
```

## 🤝 Contribuire

I contributi sono sempre benvenuti! Per contribuire:

1. Fai un fork del repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Committa le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Pusha sul branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📝 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

## 👥 Autore

- [Andrea Giummarra](https://github.com/agiummarra)

## 🙏 Ringraziamenti

- [Streamlit](https://streamlit.io/) per il framework
- [Matplotlib](https://matplotlib.org/) per la visualizzazione
- [NumPy](https://numpy.org/) per i calcoli numerici

## 📧 Contatti

Andrea Giummarra - [@agiummarra](https://github.com/agiummarra)

Link Progetto: [https://github.com/agiummarra/fractal](https://github.com/agiummarra/fractal)
