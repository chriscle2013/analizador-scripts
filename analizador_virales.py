import streamlit as st
import re
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tweepy
from textblob import TextBlob
import config  # Archivo con credenciales (crear archivo config.py con tus API keys)
import sys
if sys.version_info >= (3, 13):
    import warnings
    warnings.warn("Python 3.13 puede tener problemas de compatibilidad. Recomendamos usar 3.10-3.11")
# ======================
# 1. BASE DE DATOS DE TEMÁTICAS
# ======================
TEMATICAS = {
    # Deportes
    "Fórmula 1": {
        "palabras_clave": ["f1", "gran premio", "piloto", "carrera", "escudería"],
        "hooks": {
            "técnica": ["El {sistema} que hizo a {equipo} ganar en {circuito}"],
            "polémica": ["La decisión de la FIA que cambió el {evento}"],
            "récord": ["{piloto} rompió el récord de {marca} en {año}"]
        },
        "hashtags": ["#F1", "#Formula1"]
    },
    
    "Fútbol": {
        "palabras_clave": ["gol", "partido", "jugador", "liga", "champions"],
        "hooks": {
            "táctica": ["El {sistema_juego} que hizo campeón a {equipo}"],
            "polémica": ["El {incidente} más injusto de la historia"],
            "dato": ["{jugador} tiene este récord de {estadística}"]
        },
        "hashtags": ["#Fútbol", "#Champions"]
    },

    # Mindset
    "Mindset": {
        "palabras_clave": ["éxito", "hábitos", "mentalidad", "crecimiento"],
        "hooks": {
            "científico": ["Estudio de Harvard prueba que {hábito} aumenta {métrica}"],
            "inspiración": ["Cómo {persona} pasó de {situación} a {logro}"],
            "acción": ["Si haces esto cada mañana, tu vida cambiará en {tiempo}"]
        },
        "hashtags": ["#Mindset", "#Crecimiento"]
    },

    # Finanzas
    "Finanzas": {
        "palabras_clave": ["dinero", "inversión", "ahorro", "finanzas"],
        "hooks": {
            "impacto": ["Cómo ahorré {cantidad} en {tiempo} con {método}"],
            "error": ["El error que te hace perder {porcentaje}% de tus ingresos"],
            "sistema": ["El método {nombre} para multiplicar tu dinero"]
        },
        "hashtags": ["#Finanzas", "#Ahorro"]
    },

    # Tecnología
    "Tecnología": {
        "palabras_clave": ["robot", "ia", "tecnología", "automatización"],
        "hooks": {
            "futuro": ["Cómo {tecnología} cambiará {industria} en {año}"],
            "comparación": ["{ProductoA} vs {ProductoB}: ¿Cuál gana?"],
            "review": ["Probé {producto} y esto pasó"]
        },
        "hashtags": ["#Tecnología", "#Innovación"]
    }
}

# ======================
# 2. SISTEMA DE APRENDIZAJE AUTOMÁTICO
# ======================
class HookOptimizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = KMeans(n_clusters=5)
        self.hooks_db = []
        
    def entrenar(self, hooks_virales):
    """Entrena con hooks exitosos históricos"""
    if not hooks_virales or len(hooks_virales) < 2:
        raise ValueError("Se necesitan al menos 2 hooks para entrenar")
    
    try:
        X = self.vectorizer.fit_transform(hooks_virales)
        n_clusters = min(5, len(hooks_virales)-1)  # Asegura n_clusters < n_samples
        self.model = KMeans(n_clusters=n_clusters)
        self.model.fit(X)
        self.hooks_db = hooks_virales
    except Exception as e:
        st.error(f"Error entrenando modelo: {str(e)}")
        self.model = None
        
    def generar_hook_optimizado(self, texto, tema):
        """Genera hook basado en patrones aprendidos"""
        try:
            X_new = self.vectorizer.transform([texto])
            cluster = self.model.predict(X_new)[0]
            hooks_similares = [h for i,h in enumerate(self.hooks_db) 
                             if self.model.labels_[i] == cluster]
            
            return random.choice(hooks_similares) if hooks_similares else self.generar_hook_default(tema)
        except:
            return self.generar_hook_default(tema)
    
    def generar_hook_default(self, tema):
        """Hook base cuando no hay datos suficientes"""
        hooks = TEMATICAS.get(tema, {}).get("hooks", {}).values()
        return random.choice(list(hooks)[0]) if hooks else "Descubre esto..."

# ======================
# 3. FUNCIONES PRINCIPALES
# ======================
def analizar_tematica(texto):
    """Detecta la temática principal del texto"""
    scores = defaultdict(int)
    for tema, data in TEMATICAS.items():
        for palabra in data["palabras_clave"]:
            if re.search(rf"\b{palabra}\b", texto.lower()):
                scores[tema] += 1
    return max(scores.items(), key=lambda x: x[1]) if scores else ("General", 1)

def adaptar_formato(contenido, formato, tema):
    """Ajusta el contenido al formato de red social"""
    formatos = {
        "Reels": {"hashtags": 5, "máx_caracteres": 150},
        "TikTok": {"hashtags": 3, "máx_caracteres": 100},
        "YouTube": {"hashtags": 8, "máx_caracteres": 5000}
    }
    cfg = formatos.get(formato, formatos["Reels"])
    
    # Línea corregida (paréntesis balanceados)
    hashtags = ' '.join(random.sample(
        TEMATICAS.get(tema, {}).get("hashtags", ["#Viral"]),
        min(cfg["hashtags"], len(TEMATICAS.get(tema, {}).get("hashtags", ["#Viral"])))
    ))
    
    return f"{contenido[:cfg['máx_caracteres']]}\n\n{hashtags}"

# ======================
# 4. INTERFAZ STREAMLIT
# ======================
def main():
    st.set_page_config(layout="wide", page_title="🔥 ViralHook Generator PRO")
    
    # Inicializar sistemas
    hook_ai = HookOptimizer()
    hook_ai.entrenar([
    "Cómo ahorré $1000 en 1 mes con este método",
    "El error que el 90% comete al hacer ejercicio",
    "Comparativa: iPhone 15 vs Samsung S23 - ¿Cuál gana?",
    "Esta técnica aumentó mis ventas un 300%",
    "Lo que nadie te dice sobre invertir en cripto"
])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Configuración")
        texto = st.text_area("Pega tu contenido:", height=150)
        formato = st.selectbox("Formato:", ["Reels", "TikTok", "YouTube"])
        
    with col2:
        if st.button("🚀 Generar Contenido Viral"):
            if texto:
                # Análisis
                tema, score = analizar_tematica(texto)
                polaridad = TextBlob(texto).sentiment.polarity
                
                # Generación
                hook = hook_ai.generar_hook_optimizado(texto, tema)
                contenido = adaptar_formato(f"{hook}\n\n{texto}", formato, tema)
                
                # Resultados
                st.subheader(f"🎯 Temática: {tema} (Confianza: {score*10}%)")
                st.text_area("Contenido Optimizado:", contenido, height=300)
                
                # Métricas
                with st.expander("📊 Análisis Avanzado"):
                    st.metric("Sentimiento", "Positivo" if polaridad > 0 else "Negativo", 
                             delta=f"{polaridad:.2f}")
                    st.write(f"**Hashtags:** {contenido.splitlines()[-1]}")
            else:
                st.warning("Por favor ingresa contenido para analizar")

if __name__ == "__main__":
    main()
