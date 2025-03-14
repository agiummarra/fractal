import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def mandelbrot(h, w, max_iter, x_min, x_max, y_min, y_max):
    x, y = np.meshgrid(np.linspace(x_min, x_max, w), np.linspace(y_min, y_max, h))
    c = x + y * 1j
    z = c.copy()
    div_time = np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z * np.conj(z) > 2**2
        div_now = diverge & (div_time == 0)
        div_time[div_now] = i
        z[diverge] = 2
        
    return div_time

def julia(h, w, max_iter, x_min, x_max, y_min, y_max, c):
    x, y = np.meshgrid(np.linspace(x_min, x_max, w), np.linspace(y_min, y_max, h))
    z = x + y * 1j
    div_time = np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z * np.conj(z) > 2**2
        div_now = diverge & (div_time == 0)
        div_time[div_now] = i
        z[diverge] = 2
        
    return div_time

def sierpinski_carpet(size, iterations):
    carpet = np.ones((size, size))
    
    def cut_square(x, y, size, carpet):
        size_third = size // 3
        if size_third < 1:
            return
        
        # Taglia il quadrato centrale
        carpet[y+size_third:y+2*size_third, x+size_third:x+2*size_third] = 0
        
        # Ricorsione per gli 8 quadrati rimanenti
        for i in range(3):
            for j in range(3):
                if not (i == 1 and j == 1):  # Salta il quadrato centrale
                    cut_square(x + i * size_third, y + j * size_third, size_third, carpet)
    
    cut_square(0, 0, size, carpet)
    return carpet

def koch_snowflake(iterations, size=600):
    def koch_curve(p1, p2, iterations):
        if iterations == 0:
            return [p1, p2]
        
        # Calcola i punti intermedi
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        p3 = (p1[0] + dx/3, p1[1] + dy/3)
        p5 = (p1[0] + 2*dx/3, p1[1] + 2*dy/3)
        
        # Calcola il punto del "picco"
        angle = np.arctan2(dy, dx) + np.pi/3
        length = np.sqrt(dx**2 + dy**2) / 3
        p4 = (p3[0] + length * np.cos(angle), p3[1] + length * np.sin(angle))
        
        # Ricorsione
        curve = koch_curve(p1, p3, iterations-1)
        curve.extend(koch_curve(p3, p4, iterations-1)[1:])
        curve.extend(koch_curve(p4, p5, iterations-1)[1:])
        curve.extend(koch_curve(p5, p2, iterations-1)[1:])
        
        return curve
    
    # Inizia con un triangolo equilatero
    height = size * np.sqrt(3) / 2
    p1 = (0, 0)
    p2 = (size, 0)
    p3 = (size/2, height)
    
    # Genera le tre curve di Koch
    curve1 = koch_curve(p1, p2, iterations)
    curve2 = koch_curve(p2, p3, iterations)
    curve3 = koch_curve(p3, p1, iterations)
    
    # Combina le curve
    snowflake = curve1[:-1] + curve2[:-1] + curve3[:-1]
    
    # Estrai le coordinate x e y
    x = [p[0] for p in snowflake]
    y = [p[1] for p in snowflake]
    
    return x, y

def barnsley_fern(iterations=100000):
    # Inizializza i punti
    x, y = 0, 0
    x_points = []
    y_points = []
    
    for _ in range(iterations):
        x_points.append(x)
        y_points.append(y)
        
        # Scegli una trasformazione casuale
        r = np.random.random()
        
        if r < 0.01:  # 1% di probabilità
            x_new = 0
            y_new = 0.16 * y
        elif r < 0.86:  # 85% di probabilità
            x_new = 0.85 * x + 0.04 * y
            y_new = -0.04 * x + 0.85 * y + 1.6
        elif r < 0.93:  # 7% di probabilità
            x_new = 0.2 * x - 0.26 * y
            y_new = 0.23 * x + 0.22 * y + 1.6
        else:  # 7% di probabilità
            x_new = -0.15 * x + 0.28 * y
            y_new = 0.26 * x + 0.24 * y + 0.44
            
        x, y = x_new, y_new
        
    return x_points, y_points

def plot_fractal(fractal_type, params):
    plt.figure(figsize=(10, 10))
    
    if fractal_type == "Mandelbrot":
        h, w = params["size"], params["size"]
        max_iter = params["iterations"]
        x_min, x_max = params["x_min"], params["x_max"]
        y_min, y_max = params["y_min"], params["y_max"]
        
        fractal = mandelbrot(h, w, max_iter, x_min, x_max, y_min, y_max)
        plt.imshow(fractal, cmap=params["colormap"], extent=(x_min, x_max, y_min, y_max))
        plt.title(f"Insieme di Mandelbrot - {max_iter} iterazioni")
        plt.xlabel("Re(c)")
        plt.ylabel("Im(c)")
        
    elif fractal_type == "Julia":
        h, w = params["size"], params["size"]
        max_iter = params["iterations"]
        x_min, x_max = params["x_min"], params["x_max"]
        y_min, y_max = params["y_min"], params["y_max"]
        c = complex(params["c_real"], params["c_imag"])
        
        fractal = julia(h, w, max_iter, x_min, x_max, y_min, y_max, c)
        plt.imshow(fractal, cmap=params["colormap"], extent=(x_min, x_max, y_min, y_max))
        plt.title(f"Insieme di Julia per c={c} - {max_iter} iterazioni")
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        
    elif fractal_type == "Sierpinski Carpet":
        size = params["size"]
        iterations = params["iterations"]
        
        fractal = sierpinski_carpet(size, iterations)
        plt.imshow(fractal, cmap='binary')
        plt.title(f"Tappeto di Sierpinski - {iterations} iterazioni")
        plt.axis('off')
        
    elif fractal_type == "Koch Snowflake":
        iterations = params["iterations"]
        size = params["size"]
        
        x, y = koch_snowflake(iterations, size)
        plt.plot(x, y, 'b-')
        plt.title(f"Fiocco di Koch - {iterations} iterazioni")
        plt.axis('equal')
        plt.axis('off')
        
    elif fractal_type == "Barnsley Fern":
        iterations = params["iterations"]
        
        x, y = barnsley_fern(iterations)
        plt.scatter(x, y, s=0.2, c='green', alpha=0.5)
        plt.title(f"Felce di Barnsley - {iterations} punti")
        plt.axis('equal')
        plt.axis('off')
    
    # Converti il grafico in immagine
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def main():
    st.title("Generatore di Frattali")
    
    st.markdown("""
    ## Cos'è un frattale?
    
    Un frattale è una figura geometrica caratterizzata da un motivo che si ripete all'infinito a scale diverse, 
    mostrando autosimilarità. I frattali sono utilizzati per modellare strutture complesse in natura come nuvole, 
    montagne, coste e foglie.
    """)
    
    # Sidebar per la selezione del tipo di frattale
    fractal_type = st.sidebar.selectbox(
        "Seleziona il tipo di frattale",
        ["Mandelbrot", "Julia", "Sierpinski Carpet", "Koch Snowflake", "Barnsley Fern"]
    )
    
    # Parametri comuni
    st.sidebar.header("Parametri")
    
    params = {}
    
    if fractal_type == "Mandelbrot":
        st.markdown("""
        ### Insieme di Mandelbrot
        
        L'insieme di Mandelbrot è definito come l'insieme dei numeri complessi $c$ per cui la successione 
        $z_{n+1} = z_n^2 + c$ con $z_0 = 0$ rimane limitata.
        
        Formula: $z_{n+1} = z_n^2 + c$
        """)
        
        params["size"] = st.sidebar.slider("Dimensione", 100, 1000, 500)
        params["iterations"] = st.sidebar.slider("Iterazioni", 10, 1000, 100)
        params["x_min"] = st.sidebar.slider("X min", -2.5, 0.0, -2.0)
        params["x_max"] = st.sidebar.slider("X max", 0.0, 2.5, 1.0)
        params["y_min"] = st.sidebar.slider("Y min", -1.5, 0.0, -1.5)
        params["y_max"] = st.sidebar.slider("Y max", 0.0, 1.5, 1.5)
        params["colormap"] = st.sidebar.selectbox("Mappa colori", ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'rainbow'])
        
    elif fractal_type == "Julia":
        st.markdown("""
        ### Insieme di Julia
        
        L'insieme di Julia è definito come l'insieme dei numeri complessi $z$ per cui la successione 
        $z_{n+1} = z_n^2 + c$ (con $c$ costante) rimane limitata.
        
        Formula: $z_{n+1} = z_n^2 + c$
        """)
        
        params["size"] = st.sidebar.slider("Dimensione", 100, 1000, 500)
        params["iterations"] = st.sidebar.slider("Iterazioni", 10, 1000, 100)
        params["x_min"] = st.sidebar.slider("X min", -2.0, 0.0, -2.0)
        params["x_max"] = st.sidebar.slider("X max", 0.0, 2.0, 2.0)
        params["y_min"] = st.sidebar.slider("Y min", -2.0, 0.0, -2.0)
        params["y_max"] = st.sidebar.slider("Y max", 0.0, 2.0, 2.0)
        params["c_real"] = st.sidebar.slider("Parte reale di c", -1.0, 1.0, -0.7)
        params["c_imag"] = st.sidebar.slider("Parte immaginaria di c", -1.0, 1.0, 0.27)
        params["colormap"] = st.sidebar.selectbox("Mappa colori", ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'rainbow'])
        
    elif fractal_type == "Sierpinski Carpet":
        st.markdown("""
        ### Tappeto di Sierpinski
        
        Il tappeto di Sierpinski è un frattale che si ottiene dividendo un quadrato in 9 quadrati uguali 
        e rimuovendo quello centrale, poi ripetendo il processo per i quadrati rimanenti.
        """)
        
        params["size"] = st.sidebar.slider("Dimensione", 100, 1000, 729)  # 3^6 = 729
        params["iterations"] = st.sidebar.slider("Iterazioni", 1, 6, 4)
        
    elif fractal_type == "Koch Snowflake":
        st.markdown("""
        ### Fiocco di Koch
        
        Il fiocco di Koch è un frattale che si ottiene partendo da un triangolo equilatero e sostituendo 
        ricorsivamente il terzo centrale di ogni lato con due segmenti che formano un angolo.
        """)
        
        params["iterations"] = st.sidebar.slider("Iterazioni", 0, 6, 4)
        params["size"] = st.sidebar.slider("Dimensione", 100, 1000, 600)
        
    elif fractal_type == "Barnsley Fern":
        st.markdown("""
        ### Felce di Barnsley
        
        La felce di Barnsley è un frattale creato usando un sistema di funzioni iterate (IFS) 
        che simula la forma di una felce naturale.
        
        Formula: Quattro trasformazioni affini con probabilità diverse:
        - f₁(x,y) = (0, 0.16y) con probabilità 1%
        - f₂(x,y) = (0.85x + 0.04y, -0.04x + 0.85y + 1.6) con probabilità 85%
        - f₃(x,y) = (0.2x - 0.26y, 0.23x + 0.22y + 1.6) con probabilità 7%
        - f₄(x,y) = (-0.15x + 0.28y, 0.26x + 0.24y + 0.44) con probabilità 7%
        """)
        
        params["iterations"] = st.sidebar.slider("Punti", 10000, 200000, 50000)
    
    # Genera e mostra il frattale
    if st.sidebar.button("Genera Frattale"):
        with st.spinner("Generazione del frattale in corso..."):
            img = plot_fractal(fractal_type, params)
            st.image(img, caption=f"Frattale: {fractal_type}", use_container_width=True)

if __name__ == "__main__":
    main() 