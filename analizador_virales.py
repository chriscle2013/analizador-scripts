import streamlit as st
import re
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import sys
import spacy
from nrclex import NRCLex
import nltk

# ... (all your existing code for TEMATICAS, HookOptimizer, get_spacy_model, etc.) ...

def analizar_tematica(texto):
    # ... (your existing analizar_tematica function) ...
    scores = defaultdict(int)
    texto_lower = texto.lower() 

    for tema, data in TEMATICAS.items():
        conteo_palabras_clave = 0
        for palabra in data["palabras_clave"]:
            if re.search(r"\b" + re.escape(palabra) + r"\b", texto_lower):
                conteo_palabras_clave += 1
        
        scores[tema] = conteo_palabras_clave
    
    if not scores:
        return ("General", 0)
    
    mejor_tema = "General"
    max_puntaje = 0
    
    for tema, puntaje in scores.items():
        if puntaje > max_puntaje:
            max_puntaje = puntaje
            mejor_tema = tema
    
    confianza = min(100, max_puntaje * 20) 

    if confianza < 30 and mejor_tema != "General": 
        return ("General", 0) 

    return (mejor_tema, confianza)

def mejorar_script(script, tema, pre_generated_hook=None):
    # ... (your existing mejorar_script function) ...
    segmentos_temporales = re.findall(r"(\(\d+-\d+\s*(?:segundos|s)\).*|Escena \d+:.*)", script, re.IGNORECASE)
    tiene_estructura = bool(segmentos_temporales)
    
    mejoras_por_tema = {
        "Robótica": { 
            "transiciones": ["SFX: Sonido futurista activándose", "Corte rápido a detalle de mecanismo", "Toma de Ameca expresando una emoción sutil"],
            "logro": ["Animación de engranajes o chips", "Texto dinámico: '¡Ingeniería Maestra!'"],
            "impacto": ["Zoom dramático en la cara del robot", "Gráfico de datos en movimiento"]
        },
        "Robots Humanoides": { 
            "transiciones": ["SFX: Sonido de servos suaves", "Corte a detalle de articulación", "Toma que resalta la fluidez del movimiento", "Close-up a los ojos de Optimus"],
            "logro": ["Animación de engranajes o chips", "Texto dinámico: '¡Ingeniería Maestra!'"],
            "impacto": ["Zoom dramático en la cara del robot", "Gráfico de datos en movimiento", "Montaje de aplicaciones diversas del robot"]
        },
        "Fútbol": {
            "transiciones": ["SFX: Hinchada rugiendo", "Slow motion de jugada clave", "Gráfico animado de estadística de jugador"],
            "logro": ["Repetición en cámara lenta del gol", "Gráfico de 'heatmap' de la cancha"]
        },
        "Finanzas": {
            "transiciones": ["Gráfico animado de crecimiento/caída", "Zoom a cifras clave", "SFX: Sonido de calculadora o transacción"],
            "logro": ["Gráfico de barra de crecimiento", "Montaje de billetes o monedas"]
        },
        "Mascotas": {
            "transiciones": ["SFX: Sonido de risas o asombro", "Corte a cara de sorpresa del dueño", "Música divertida subiendo", "Primer plano a la expresión traviesa de la mascota"],
            "consejo": ["Lista de consejos en pantalla", "Demostración visual de la solución"]
        },
        "Fórmula 1": {
            "transiciones": ["SFX: Chirrido de neumáticos", "Cámara lenta del trompo", "Toma en cabina del piloto reaccionando", "Corte rápido entre diferentes ángulos de la acción"],
            "logro": ["Gráfico de tiempos de vuelta subiendo a P1", "Celebración en el pit wall", "Cámara lenta del cruce de meta"], 
            "velocidad": ["Efecto de velocidad en el coche", "Onboard a toda velocidad"], 
            "pole": ["Tabla de tiempos resaltando P1", "Onboard de vuelta clasificatoria", "Toma en cabina del piloto reaccionando"], 
        }
    }
    
    reemplazos_genericos = {
        "{numero}": str(random.randint(10, 60)),
        "{cantidad}": str(random.randint(1, 10)),
        "{pregunta}": "¿Qué te pareció?", 
    }
    
    plantillas_genericas = {
        "mejora_visual": [
            "💡 PRO TIP: Usa primeros planos cada 3 segundos",
            "🎬 TÉCNICA: Cambio de ángulo tras cada afirmación"
        ],
        "llamado_accion": [
            "👇 ¿Qué opinas? Comenta '{pregunta}'",
            "🔥 No te pierdas más contenido como este → @tu_canal",
            "✅ ¡Síguenos para más!"
        ]
    }

    script_final_mejorado = []

    if tiene_estructura:
        lineas = script.split('\n')
        
        for i, linea in enumerate(lineas):
            script_final_mejorado.append(linea) 
            
            if re.search(r"^\(\d+-\d+\s*(?:segundos|s)\)", linea, re.IGNORECASE) or re.search(r"^Escena \d+:", linea, re.IGNORECASE):
                mejora_opciones = mejoras_por_tema.get(tema, {}).get("transiciones")
                
                if tema == "Fórmula 1":
                    if re.search(r'\b(pole|q3|última vuelta|verstappen)\b', linea.lower()):
                        mejora_opciones = mejoras_por_tema["Fórmula 1"].get("pole", mejoras_por_tema["Fórmula 1"].get("logro"))
                    elif re.search(r'\b(trompos|spin)\b', linea.lower()):
                        mejora_opciones = mejoras_por_tema["Fórmula 1"].get("transiciones") 
                elif tema == "Robots Humanoides" or tema == "Robótica": 
                    if re.search(r'\b(precisión|eficiencia|movilidad|diseñado|optimus)\b', linea.lower()):
                         mejora_opciones = mejoras_por_tema["Robots Humanoides"].get("impacto", mejoras_por_tema["Robots Humanoides"].get("transiciones"))


                if mejora_opciones:
                    mejora = random.choice(mejora_opciones)
                else: 
                    mejora = random.choice(plantillas_genericas["mejora_visual"])
                
                for k, v in reemplazos_genericos.items():
                    mejora = mejora.replace(k, v)
                
                if not mejora.strip().startswith("✨ MEJORA:"):
                    script_final_mejorado.append(f"✨ MEJORA: {mejora}")
                else:
                    script_final_mejorado.append(mejora) 
                
        cta_already_present_in_original = any(re.search(r"(comenta|suscribe|siguenos|cta|subscribe)", l.lower()) for l in script.split('\n')[-7:])

        if not cta_already_present_in_original:
            llamado_accion_gen = random.choice(plantillas_genericas["llamado_accion"])
            for k, v in reemplazos_genericos.items():
                llamado_accion_gen = llamado_accion_gen.replace(k, v)
            script_final_mejorado.append(f"\n(FINAL) 📲 LLAMADA A LA ACCIÓN: {llamado_accion_gen}")
            script_final_mejorado.append(f"✨ SUGERENCIA VISUAL: Considera añadir cortes rápidos y música dinámica.")
            
    else: 
        gancho_a_usar = pre_generated_hook if pre_generated_hook else generar_hook(tema, reemplazos_genericos)
        llamado_accion_gen = random.choice(plantillas_genericas["llamado_accion"])
        for k, v in reemplazos_genericos.items():
            llamado_accion_gen = llamado_accion_gen.replace(k, v)
        
        script_final_mejorado.append(f"(0-5 segundos) 🎯 GANCHO INICIAL: {gancho_a_usar}")
        script_final_mejorado.append("\n" + script.strip() + "\n") 
        script_final_mejorado.append(f"(FINAL) 📲 LLAMADA A LA ACCIÓN: {llamado_accion_gen}")
        script_final_mejorado.append(f"✨ SUGERENCIA VISUAL: Considera añadir cortes rápidos y música dinámica.")

    script_final = '\n'.join(script_final_mejorado)
    
    hashtags = TEMATICAS.get(tema, {}).get("hashtags", ["#Viral", "#Trending"])
    script_final += f"\n\n🔖 HASHTAGS: {' '.join(hashtags[:4])}" 
    
    return script_final

def generar_hook(tema, reemplazos):
    # ... (your existing generar_hook function) ...
    hooks_tema = TEMATICAS.get(tema, {}).get("hooks", {})
    hooks_genericos = {
        "impacto": ["Lo que nadie te dijo sobre {tema}"],
        "curiosidad": ["¿Por qué {tema} está revolucionando todo?"],
        "pregunta": ["¿Estás listo para {tema}?"],
    }
    
    hooks_disponibles = []
    if hooks_tema:
        for estrategia in hooks_tema.values():
            hooks_disponibles.extend(estrategia)
    
    for hook_gen_list in hooks_genericos.values():
        hooks_disponibles.extend(hook_gen_list)
    
    hook = random.choice(hooks_disponibles) if hooks_disponibles else "Descubre esto que cambiará tu perspectiva"
    
    for k, v in reemplazos.items():
        hook = hook.replace(k, v)
    
    return hook

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except nltk.downloader.DownloadError:
        nltk.download('averaged_perceptron_tagger')
    
def main():
    st.set_page_config(layout="wide", page_title="🔥 ViralHook Generator PRO")
    
    download_nltk_data()

    global nlp 
    nlp = get_spacy_model()

    hook_ai = HookOptimizer()
    hook_ai.entrenar([
        "Cómo los robots como Ameca están cambiando la industria",
        "La evolución de los humanoides en 2024",
        "Ameca vs humanos: ¿Quién es más expresivo?",
        "Esta tecnología robótica te sorprenderá",
        "El secreto para que tu perro deje de ladrar",
        "Las travesuras más épicas de gatos en casa",
        "¿Por qué este golden retriever es viral?",
        "¡Los 5 trompos más locos de la F1 en Silverstone!",
        "La verdad sobre el rendimiento de Ferrari en F1",
        "El error de Hamilton que le costó la carrera",
        "Max Verstappen se llevó la pole en el último segundo en Silverstone", 
        "El gato más destructor de cajas del mundo", 
        "Optimus de Tesla: el robot que revoluciona las fábricas",
        "Mira a Optimus haciendo esto en el laboratorio de Tesla", 
        "La precisión de Optimus en tareas delicadas" 
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎬 Script para Analizar")
        
        # --- FIX: Ensure script_content is initialized in session_state ---
        if 'script_content' not in st.session_state:
            st.session_state.script_content = ""

        texto = st.text_area("Pega tu script completo:", 
                             height=300,
                             placeholder="Ej: (0-3 segundos) Video impactante...",
                             key="script_input_area", # Usa un key único
                             value=st.session_state.script_content) # Use the session_state value
        
        # Botón para borrar el contenido
        if st.button("🗑️ Borrar Script", key="clear_script_button"):
            st.session_state.script_content = "" # Resetea el valor en el estado de sesión
            st.experimental_rerun() # Fuerza una recarga para que el text_area se vacíe

    with col2:
        # The 'texto' variable is now guaranteed to be assigned a value from st.text_area
        # even if it's an empty string. So 'if texto:' is safe to use here.
        if st.button("🚀 Optimizar Contenido"):
            if texto: # This check is now safe
                with st.spinner("Analizando y mejorando..."):
                    tema, confianza = analizar_tematica(texto)
                    blob = TextBlob(texto) 
                    polaridad = blob.sentiment.polarity
                    
                    generated_hook = hook_ai.generar_hook_optimizado(texto, tema) 
                    
                    script_mejorado = mejorar_script(texto, tema, pre_generated_hook=generated_hook)
                    
                    hashtags_display = ' '.join(TEMATICAS.get(tema, {}).get("hashtags", ["#Viral"]))

                    st.subheader(f"🎯 Temática: **{tema}** (Confianza: **{confianza}%**)")
                    
                    st.text_area("Hook Viral Recomendado:", value=generated_hook, height=100) 
                    
                    st.text_area("Script Optimizado:", value=script_mejorado, height=450) 
                    
                    with st.expander("📊 Análisis Avanzado"):
                        st.metric("Sentimiento General",
                                  "🔥 Positivo" if polaridad > 0.1 else "😐 Neutral" if polaridad > -0.1 else "⚠️ Negativo",
                                  delta=f"{polaridad:.2f}")

                        emotions = NRCLex(texto).affect_frequencies
                        st.subheader("Emociones Detectadas:")
                        emociones_relevantes = {k: v for k, v in emotions.items() if v > 0.05} 
                        if emociones_relevantes:
                            for emotion, freq in sorted(emociones_relevantes.items(), key=lambda item: item[1], reverse=True):
                                st.write(f"- **{emotion.capitalize()}**: {freq:.2%}")
                        else:
                            st.write("No se detectaron emociones fuertes en el script.")

                        st.write(f"🔍 Hashtags recomendados: {hashtags_display}")
            else:
                st.warning("Por favor ingresa un script para analizar")

if __name__ == "__main__":
    main()
