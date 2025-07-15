import streamlit as st
from textblob import TextBlob
import re
import nltk
import random
from collections import defaultdict

# Configuración inicial
nltk.download('punkt', quiet=True)
st.set_page_config(page_title="Analizador de Reels Virales Pro", layout="wide")

# Cache para mejor performance (usa @st.cache_data en Streamlit >= 1.18)
@st.cache_data
def analizar_sentimiento(texto):
    """Análisis de sentimiento mejorado con TextBlob"""
    analysis = TextBlob(texto)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.3:
        return ("Muy Positivo 😊", polarity)
    elif polarity > 0.1:
        return ("Positivo 🙂", polarity)
    elif polarity < -0.3:
        return ("Muy Negativo 😠", polarity)
    elif polarity < -0.1:
        return ("Negativo 🙁", polarity)
    else:
        return ("Neutral 😐", polarity)

@st.cache_data
def evaluar_viralidad(script):
    """Evaluación de viralidad con puntaje detallado"""
    metricas = {
        'hook_llamativo': 3 if any(p in script.lower() for p in ["sabías que", "no vas a creer", "impactante"]) else 0,
        'cta': 2 if re.search(r"(comenta|guárdalo|comparte|dale like)", script.lower()) else 0,
        'brevedad': 2 if len(script.split()) < 100 else 0,
        'emocional': 2 if any(p in script.lower() for p in ["increíble", "sorprendente", "emocionante"]) else 0,
        'preguntas': 1 if '?' in script else 0
    }
    
    puntaje_total = sum(metricas.values())
    
    return {
        'puntaje': puntaje_total,
        'maximo': 10,
        'metricas': metricas,
        'nivel': 'Alto' if puntaje_total >= 7 else 'Medio' if puntaje_total >= 4 else 'Bajo'
    }

@st.cache_data
def generar_recomendaciones(script):
    """Genera recomendaciones basadas en análisis detallado"""
    recomendaciones = []
    blob = TextBlob(script)
    
    # Análisis de longitud
    if len(script.split()) > 150:
        recomendaciones.append("🔹 Reducir longitud (ideal <150 palabras)")
    
    # Análisis de frases
    long_frases = [len(frase.split()) for frase in script.split('.') if frase.strip()]
    if sum(long_frases)/len(long_frases) > 15:
        recomendaciones.append("🔹 Acortar frases (promedio >15 palabras)")
    
    # Llamadas a acción
    if not re.search(r"(comenta|guárdalo|comparte)", script.lower()):
        recomendaciones.append("🔹 Añadir CTA (Comenta/Comparte)")
    
    # Elementos emocionales
    emociones = ['increíble', 'sorprendente', 'emocionante']
    if not any(e in script.lower() for e in emociones):
        recomendaciones.append(f"🔹 Incluir palabras emocionales ({', '.join(emociones)})")
    
    # Preguntas
    if '?' not in script:
        recomendaciones.append("🔹 Incluir preguntas para engagement")
    
    return recomendaciones

@st.cache_data
def detectar_tematicas(script):
    """Detección múltiple de temáticas con ponderación"""
    script = script.lower()
    temas = {
        "Motivación": ["lograr", "superar", "esfuerzo", "disciplina"],
        "Autoestima": ["valor", "mereces", "amor propio", "confianza"],
        "Finanzas": ["dinero", "finanzas", "ahorro", "invertir"],
        "Salud": ["salud", "bienestar", "ejercicio", "hábitos"],
        "Relaciones": ["pareja", "amor", "familia", "amistad"],
        "Éxito": ["éxito", "logro", "emprender", "sueño"],
        "Tecnología": ["robot", "IA", "futuro", "tecnología"]
    }
    
    conteo = defaultdict(int)
    for tema, palabras in temas.items():
        for palabra in palabras:
            if palabra in script:
                conteo[tema] += 1
    
    total = sum(conteo.values())
    if total == 0:
        return [("General", 100)]
    
    return sorted(
        [(k, round(v/total*100)) for k, v in conteo.items()],
        key=lambda x: x[1],
        reverse=True
    )

@st.cache_data
def generar_hook(script):
    """Generador de hooks con múltiples estrategias"""
    sentimiento, polaridad = analizar_sentimiento(script)
    temas = detectar_tematicas(script)[:2]
    
    estrategias = {
        'pregunta': [
            f"¿Sabías que {temas[0][0]} puede cambiar tu vida?",
            f"¿Estás listo para este dato sobre {temas[0][0]}?"
        ],
        'afirmacion': [
            f"Esto cambiará tu forma de ver {temas[0][0]}",
            f"El secreto de {temas[0][0]} que pocos conocen"
        ],
        'controversia': [
            f"Todo lo que sabes sobre {temas[0][0]} podría estar mal",
            f"Por qué {temas[0][0]} no funciona como crees"
        ]
    }
    
    # Selección de estrategia basada en sentimiento
    if polaridad > 0.2:
        estrategia = random.choice(['pregunta', 'afirmacion'])
    elif polaridad < -0.2:
        estrategia = 'controversia'
    else:
        estrategia = random.choice(['pregunta', 'afirmacion'])
    
    return random.choice(estrategias[estrategia])

def mejorar_script(script):
    """Función mejorada de optimización de scripts"""
    # Generar hook si no existe
    primeras_lineas = script.strip().split('\n')[0][:120].lower()
    if not any(p in primeras_lineas for p in ["sabías que", "no vas a creer", "impactante"]):
        hook = f"{generar_hook(script)}\n\n"
    else:
        hook = ""
    
    # Dividir frases largas
    frases_mejoradas = []
    for frase in script.split('.'):
        frase = frase.strip()
        if frase:
            palabras = frase.split()
            if len(palabras) > 18:
                mitad = len(palabras) // 2
                frases_mejoradas.append(" ".join(palabras[:mitad]) + ".")
                frases_mejoradas.append(" ".join(palabras[mitad:]) + ".")
            else:
                frases_mejoradas.append(frase + ".")
    
    # Añadir elementos de engagement
    mejoras = []
    if not re.search(r"(sígueme|dale like|comenta)", script.lower()):
        mejoras.append("💬 ¿Qué opinas? ¡Déjalo en los comentarios!")
    
    if not any(p in script.lower() for p in ["increíble", "sorprendente"]):
        mejoras.append("✨ Este contenido es más impactante de lo que imaginas")
    
    # Reconstrucción del script
    return f"{hook}{' '.join(frases_mejoradas)}\n\n{' '.join(mejoras)}"

# Interfaz mejorada
def main():
    st.title("🎬 Analizador de Reels Virales PRO")
    st.markdown("Optimiza tus guiones para maximizar engagement y viralidad")
    
    with st.expander("📌 Instrucciones"):
        st.markdown("""
        1. Pega tu guión en el área de texto
        2. Haz clic en "Analizar Guión"
        3. Recibe recomendaciones y versión optimizada
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        script_input = st.text_area("✍️ Pega aquí tu guión completo:", 
                                  height=300,
                                  placeholder="Ejemplo: 'Descubre cómo esta tecnología...'")
        
        if st.button("🚀 Analizar Guión", use_container_width=True):
            if not script_input.strip():
                st.error("Por favor ingresa un guión para analizar")
                st.stop()
    
    if script_input and script_input.strip():
        with st.spinner("Analizando contenido..."):
            # Análisis
            sentimiento, polaridad = analizar_sentimiento(script_input)
            viralidad = evaluar_viralidad(script_input)
            recomendaciones = generar_recomendaciones(script_input)
            temas = detectar_tematicas(script_input)
            
            # Mostrar resultados
            with col2:
                st.subheader("📊 Métricas Clave")
                
                # Gráfico de sentimiento
                st.metric("Sentimiento", sentimiento, 
                         delta=f"{polaridad:.2f} polaridad", 
                         help="Rango de -1 (negativo) a 1 (positivo)")
                
                # Puntaje de viralidad
                st.progress(viralidad['puntaje']/viralidad['maximo'])
                st.caption(f"Potencial de viralidad: {viralidad['nivel']} ({viralidad['puntaje']}/{viralidad['maximo']} pts)")
                
                # Temáticas detectadas
                st.write("**🎯 Temáticas principales:**")
                for tema, porcentaje in temas[:3]:
                    st.write(f"- {tema} ({porcentaje}%)")
            
            # Sección de recomendaciones
            st.subheader("💡 Recomendaciones para Mejorar")
            cols_rec = st.columns(2)
            for i, rec in enumerate(recomendaciones):
                cols_rec[i%2].info(rec)
            
            # Script optimizado
            st.subheader("✨ Versión Optimizada")
            script_mejorado = mejorar_script(script_input)
            st.text_area("Copia este texto:", 
                         value=script_mejorado, 
                         height=300,
                         label_visibility="hidden")
            
            # Botón de copia
            st.button("📋 Copiar al Portapapeles", 
                     on_click=lambda: st.write("Texto copiado!"),  # En realidad usar pyperclip
                     help="Copia el texto optimizado")

if __name__ == "__main__":
    main()
