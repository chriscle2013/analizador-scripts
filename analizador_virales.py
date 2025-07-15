import streamlit as st
from textblob import TextBlob
import re
import nltk
import random
from collections import defaultdict

# Configuración inicial
nltk.download('punkt', quiet=True)
st.set_page_config(page_title="Optimizador de Scripts Virales PRO", layout="wide")

# --- Base de Datos de Temáticas y Hooks Virales ---
TEMATICAS = {
    "Tecnología": {
        "palabras_clave": ["robot", "IA", "tecnología", "humanoide", "automatización", "futurista"],
        "hooks": [
            "¿Estás listo para esta REVOLUCIÓN tecnológica?",
            "Esta actualización de IA te dejará BOQUIABIERTO",
            "El futuro que ves en películas YA ESTÁ AQUÍ",
            "Ningún humano puede hacer esto...",
            "Advertencia: Esto cambiará tu visión de la tecnología"
        ]
    },
    "Deportes (Fórmula 1)": {
        "palabras_clave": ["f1", "formula 1", "carrera", "piloto", "gran premio"],
        "hooks": [
            "El SECRETO que los equipos de F1 no quieren que sepas",
            "¿Por qué este adelanto CAMBIARÁ la F1 para siempre?",
            "Así se ROMPIERON todas las reglas en la última carrera",
            "Lo que nunca viste en las cámaras de la F1",
            "El movimiento ILEGAL que ganó el Gran Premio"
        ]
    },
    "Deportes (Fútbol)": {
        "palabras_clave": ["fútbol", "gol", "partido", "jugador", "liga"],
        "hooks": [
            "La JUGADA que dejó a todos EN SHOCK",
            "¿Sabías que este equipo usa TÁCTICAS PROHIBIDAS?",
            "El ERROR que costó MILLONES al club",
            "Así se ENTRENA como los grandes campeones",
            "Lo que NUNCA te muestran en el vestuario"
        ]
    },
    "Mindset": {
        "palabras_clave": ["mente", "éxito", "hábitos", "mentalidad", "crecimiento"],
        "hooks": [
            "El HÁBITO MATUTINO de todos los millonarios",
            "Tu mente es tu PRISIÓN... Así puedes LIBERARTE",
            "La VERDAD sobre el éxito que nadie te dice",
            "Si haces esto cada mañana, tu vida CAMBIARÁ",
            "Científicamente comprobado: Este método FUNCIONA"
        ]
    },
    "Marketing Digital": {
        "palabras_clave": ["marketing", "redes", "ventas", "conversión", "lead"],
        "hooks": [
            "La ESTRATEGIA BANEADA que genera leads",
            "Así DOBLÉ mis ventas en 7 días",
            "El SECRETO que las agencias no quieren revelar",
            "Cómo VENDER sin que te odien",
            "Este truco de $0 te pondrá en el TOP 1%"
        ]
    },
    "Automatización": {
        "palabras_clave": ["automatizar", "bot", "flujo", "proceso", "ahorro"],
        "hooks": [
            "Cómo AUTOMATICÉ mi negocio y me liberé 20h/semana",
            "El BOT que hace tu trabajo mientras duermes",
            "Advertencia: Esto dejará OBSOLETOS muchos trabajos",
            "Automatiza esto y GANA 3 horas diarias",
            "La herramienta SECRETA que uso para todo"
        ]
    }
}

# --- Funciones Mejoradas ---
def detectar_tematica(script):
    script = script.lower()
    conteo = defaultdict(int)
    
    for tema, data in TEMATICAS.items():
        for palabra in data["palabras_clave"]:
            if palabra in script:
                conteo[tema] += 1
    
    if not conteo:
        return "General"
    
    return max(conteo.items(), key=lambda x: x[1])[0]

def generar_hook(script):
    tema = detectar_tematica(script)
    sentimiento, _ = analizar_sentimiento(script)
    
    # Hooks específicos por temática
    if tema in TEMATICAS:
        hooks = TEMATICAS[tema]["hooks"]
    else:
        hooks = [
            "Esto cambiará tu perspectiva",
            "¿Sabías que esto puede transformar todo?",
            "La verdad que necesitas escuchar"
        ]
    
    # Filtrado por sentimiento
    hooks_filtrados = []
    for hook in hooks:
        if (sentimiento == "Positivo" and "!" in hook) or \
           (sentimiento == "Negativo" and ("?" in hook or "Advertencia" in hook)) or \
           sentimiento == "Neutral":
            hooks_filtrados.append(hook)
    
    return random.choice(hooks_filtrados) if hooks_filtrados else random.choice(hooks)

@st.cache_data
def analizar_sentimiento(texto):
    analysis = TextBlob(texto)
    polarity = analysis.sentiment.polarity
    if polarity > 0.3: return ("Positivo", polarity)
    elif polarity < -0.1: return ("Negativo", polarity)
    else: return ("Neutral", polarity)

def mejorar_script(script):
    # Conservar estructura temporal
    segmentos = re.split(r'(\(\d+-\d+ segundos\))', script)
    
    # Generar hook alineado
    hook = generar_hook(script)
    if not any(p in script.lower() for p in ["sabías que", "no vas a creer", "impactante"]):
        segmentos[0] = f"{hook}\n\n{segmentos[0]}"
    
    # Procesar cada segmento
    for i in range(1, len(segmentos), 2):
        segmentos[i+1] = procesar_segmento(segmentos[i+1])
    
    return formatear_output(''.join(segmentos))

def procesar_segmento(segmento):
    # Optimización de frases
    frases = []
    for frase in segmento.split('.'):
        frase = frase.strip()
        if frase:
            # Dividir frases largas
            palabras = frase.split()
            if len(palabras) > 18:
                mitad = len(palabras) // 2
                frases.append(" ".join(palabras[:mitad]) + ".")
                frases.append(" ".join(palabras[mitad:]) + ".")
            else:
                frases.append(frase + ".")
    
    # Añadir elementos virales
    tema = detectar_tematica(segmento)
    if tema in TEMATICAS and random.random() > 0.5:
        frases.append(random.choice([
            "💥 Esto es solo el comienzo...",
            "🚀 ¿Qué opinas? ¡Déjalo en los comentarios!",
            f"🔔 Sígueme para más sobre {tema.lower()}"
        ]))
    
    return ' '.join(frases)

def formatear_output(script):
    # Mejorar formato visual
    script = re.sub(r'\((\d+-\d+ segundos)\)', r'\n\n### \1\n', script)
    script = re.sub(r'\.\s+', '.\n\n', script)
    return script.strip()

# --- Interfaz Mejorada ---
def main():
    st.title("🚀 Generador de Scripts Virales")
    st.markdown("Potencia tus videos con hooks diseñados para viralizarse")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        script = st.text_area("Pega tu script aquí:", height=300,
                            placeholder="Ejemplo: 'La nueva actualización del robot Ameca...'")
        
    with col2:
        if st.button("Generar Versión Viral"):
            if script:
                with st.spinner("Optimizando para máxima viralidad..."):
                    # Análisis
                    tema = detectar_tematica(script)
                    sentimiento, polaridad = analizar_sentimiento(script)
                    
                    # Mejora
                    script_mejorado = mejorar_script(script)
                    
                    # Resultados
                    st.success(f"Temática detectada: {tema}")
                    st.text_area("Script Optimizado:", script_mejorado, height=300)
                    
                    # Estadísticas
                    with st.expander("📊 Métricas Avanzadas"):
                        st.metric("Potencial Viral", "Alto" if polaridad > 0.2 else "Medio", 
                                 delta=f"{polaridad:.2f} polaridad")
                        st.progress(min(max((polaridad + 1)/2, 0.1), 
                                   text="Engagement Estimado")
            else:
                st.error("Por favor ingresa un script para optimizar")

if __name__ == "__main__":
    main()
