import streamlit as st
import re
import random
from datetime import datetime
from collections import defaultdict
import pytz

# 1. BASE DE DATOS MULTIDIMENSIONAL (30+ TEMÁTICAS)
TEMATICAS = {
    # Tecnología
    "Inteligencia Artificial": {
        "palabras_clave": ["IA", "machine learning", "red neuronal", "chatbot", "GPT"],
        "hooks": {
            "pregunta": ["¿Sabías que {IA} puede {acción}? Esto cambiará {industria}", 
                        "¿Estamos preparados para {impacto_IA}?"],
            "impacto": ["{avance_IA} que dejó obsoletos a los {profesión}"],
            "controversia": ["Lo que {empresa_tec} no quiere que sepas sobre {tema_tec}"]
        },
        "ctas": ["¿La IA debería regularse? Debate ↓ #IAEthics", 
                "¿Reemplazará tu trabajo? Comenta 👇"],
        "hashtags": ["#IA", "#FuturoTecnológico"]
    },
    
    # Deportes
    "Fútbol": {
        "palabras_clave": ["gol", "partido", "jugador", "liga", "tarjeta", "árbitro"],
        "hooks": {
            "táctica": ["El {sistema_juego} que hizo campeón a {equipo}"],
            "polémica": ["El {incidente} más injusto de la historia"],
            "récord": ["{jugador} rompió este récord de {estadística}"]
        },
        "ctas": ["¿Quién es el mejor? Vota →", "¿Penal o no? Juzga tú ↓"],
        "hashtags": ["#Fútbol", "#LaPulga"]
    },

    # Salud
    "Fitness": {
        "palabras_clave": ["ejercicio", "gimnasio", "proteína", "rutina"],
        "hooks": {
            "resultados": ["{rutina} para {objetivo} en {tiempo}"],
            "mitos": ["El error que arruina tu {ejercicio}"],
            "ciencia": ["Estudio prueba que {hábito} quema {porcentaje}% más grasa"]
        },
        "ctas": ["¿Probaste esto? Cuéntanos ↓", "Guarda para después 💾"],
        "hashtags": ["#Fitness", "#GymLife"]
    }
}

# 2. GENERADOR DE EJEMPLOS CONTEXTUALES
EJEMPLOS = {
    "IA": ["GPT-4", "DeepMind"],
    "acción": ["escribir código", "diagnosticar cáncer", "componer música"],
    "industria": ["la medicina", "el arte", "tu trabajo"],
    "equipo": ["el Barcelona", "Argentina", "el Real Madrid"],
    "sistema_juego": ["tiki-taka", "contraataque", "presión alta"],
    "rutina": ["este entrenamiento", "esta secuencia", "esta combinación"],
    "objetivo": ["ganar músculo", "quemar grasa", "definir"]
}

# 3. DETECTOR DE TENDENCIAS ESTACIONALES
def detectar_tendencias():
    mx_tz = pytz.timezone('America/Mexico_City')
    hoy = datetime.now(mx_tz)
    tendencias = []
    
    # Eventos globales
    if hoy.month == 12: tendencias.append(("Navidad", ["regalo", "celebración", "familia"]))
    if hoy.month == 6 and hoy.day >= 15: tendencias.append(("Mundial", ["gol", "selección", "partido"]))
    
    # Tecnología (Ej: lanzamientos conocidos)
    if hoy.month == 9: tendencias.append(("Apple Event", ["iPhone", "iOS", "keynote"]))
    
    return tendencias

# 4. GENERADOR DE HOOKS INTELIGENTE
def generar_hook(tema, texto):
    # Combinar palabras clave de tema + tendencias
    palabras_clave = TEMATICAS[tema]["palabras_clave"].copy()
    for tendencia, palabras in detectar_tendencias():
        palabras_clave.extend(palabras)
    
    # Seleccionar estrategia basada en sentimiento
    blob = TextBlob(texto)
    polaridad = blob.sentiment.polarity
    
    if polaridad > 0.3:
        estrategia = random.choice(["pregunta", "impacto"])
    elif polaridad < -0.1:
        estrategia = "controversia"
    else:
        estrategia = random.choice(list(TEMATICAS[tema]["hooks"].keys()))
    
    # Generar hook con plantillas dinámicas
    plantilla = random.choice(TEMATICAS[tema]["hooks"][estrategia])
    hook = plantilla
    
    # Rellenar placeholders
    for placeholder in re.findall(r"\{(.*?)\}", plantilla):
        if placeholder in EJEMPLOS:
            hook = hook.replace("{"+placeholder+"}", random.choice(EJEMPLOS[placeholder]))
    
    # Añadir emojis según temática
    emojis = {
        "IA": "🤖", "Fútbol": "⚽", "Fitness": "💪"
    }.get(tema, "✨")
    
    return f"{emojis} {hook}"

# 5. SISTEMA MULTIFORMATO
def adaptar_formato(contenido, formato):
    formatos = {
        "Reels": {
            "estructura": "Hook (0-3s) → Desarrollo (15s) → CTA (2s)",
            "hashtags": 5,
            "máx_caracteres": 150
        },
        "TikTok": {
            "estructura": "Impacto inmediato + Baile/Reto opcional",
            "hashtags": 3,
            "máx_caracteres": 100
        },
        "YouTube": {
            "estructura": "Preview + Intro + Contenido + CTA extendido",
            "hashtags": 8,
            "máx_caracteres": 5000
        }
    }
    
    cfg = formatos[formato]
    contenido = contenido[:cfg["máx_caracteres"]]
    hashtags = ' '.join(random.sample(TEMATICAS[tema]["hashtags"], min(cfg["hashtags"], len(TEMATICAS[tema]["hashtags"])))
    
    return f"{contenido}\n\n{hashtags}"

# 6. INTERFAZ STREAMLIT (MEJORADA)
def main():
    st.set_page_config(layout="wide", page_title="🔥 ViralHook Generator PRO")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Configuración")
        texto = st.text_area("Pega tu contenido:", height=200)
        formato = st.selectbox("Formato:", ["Reels", "TikTok", "YouTube"])
        tema_manual = st.selectbox("O elige temática:", list(TEMATICAS.keys()))
        
    with col2:
        if st.button("🚀 Generar Contenido Viral"):
            if texto:
                # Análisis automático
                tema, _ = analizar_tematica(texto)
                tema = tema if tema != "General" else tema_manual
                
                # Generación de contenido
                hook = generar_hook(tema, texto)
                cta = random.choice(TEMATICAS[tema]["ctas"])
                contenido = adaptar_formato(f"{hook}\n\n{texto}\n\n{cta}", formato)
                
                # Mostrar resultados
                st.subheader(f"🎯 Temática: {tema}")
                st.text_area("Contenido optimizado:", contenido, height=300)
                
                # Insights
                with st.expander("🔍 Análisis Avanzado"):
                    st.write(f"**Estrategia:** {estrategia.capitalize()}")
                    st.write(f"**Tendencias aplicadas:** {', '.join(t[0] for t in detectar_tendencias())}")
                    st.progress(min(int((polaridad+1)*50), 100), text="Potencial Viral")
            else:
                st.warning("Por favor ingresa contenido")

if __name__ == "__main__":
    main()
