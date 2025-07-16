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

    # Robótica
    "Robots Humanoides": {
        "palabras_clave": ["humanoide", "bípedo", "androide", "atlas", "asimo", "ameca", "engineered arts"],
        "hooks": {
            "técnica": ["Los desafíos de la **locomoción bípeda** en {nombre_robot}"],
            "aplicación": ["Cómo los humanoides están revolucionando la {industria}"],
            "avance": ["El nuevo sensor de {compañía} que permite a los humanoides {acción_mejorada}"]
        },
        "hashtags": ["#Humanoides", "#RobotsHumanoides", "#Bípedos", "#Ameca"]
    },
    "Inteligencia Artificial en Robótica": {
        "palabras_clave": ["ia", "aprendizaje automático", "machine learning", "visión artificial", "deep learning", "algoritmos"],
        "hooks": {
            "técnica": ["La **red neuronal** que permite a {robot} reconocer {objeto}"],
            "aplicación": ["IA para la **navegación autónoma** en {entorno_complejo}"],
            "impacto": ["Cómo el {algoritmo_ia} está optimizando el {proceso_robotico}"]
        },
        "hashtags": ["#AIRobótica", "#IA", "#MachineLearningRobots", "#VisiónArtificial"]
    },
    "Robots Colaborativos (Cobots)": {
        "palabras_clave": ["cobots", "colaborativos", "seguridad", "interacción h-r", "industria 4.0"],
        "hooks": {
            "beneficio": ["**Cobots**: Mejorando la {productividad} y la {seguridad} en {sector}"],
            "implementación": ["Desafíos y soluciones al integrar **cobots** en {tipo_empresa}"],
            "futuro": ["El rol de los **robots colaborativos** en la {próxima_década}"]
        },
        "hashtags": ["#Cobots", "#RobotsColaborativos", "#Industria40"]
    },
    "Robótica Médica": {
        "palabras_clave": ["cirugía robótica", "quirúrgico", "da vinci", "rehabilitación", "exosqueletos", "telemedicina"],
        "hooks": {
            "innovación": ["**Robótica médica**: La precisión de {sistema_robotico} en {procedimiento_medico}"],
            "impacto_paciente": ["Cómo los **exosqueletos** están transformando la {condicion_paciente}"],
            "futuro": ["La próxima generación de **robots asistenciales** en {ámbito_salud}"]
        },
        "hashtags": ["#RobóticaMédica", "#CirugíaRobótica", "#Exoesqueletos", "#SaludDigital"]
    },

    # Mascotas
    "Mascotas": {
        "palabras_clave": ["perro", "gato", "hámster", "pájaro", "mascota", "animales", "cachorros", "golden retriever", "labrador", "loro", "periquito"],
        "hooks": {
            "humor": ["Tu mascota también hace ESTO para volverte loco", "¿Listo para reírte? Las travesuras más épicas de {animal}"],
            "consejo": ["El secreto para que tu {tipo_mascota} deje de {mal_hábito}"],
            "emocional": ["La historia de {animal} que te derretirá el corazón"]
        },
        "hashtags": ["#Mascotas", "#AnimalesGraciosos", "#MascotasVirales", "#PetsOfTikTok"]
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

    # Tecnología (General)
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
# 2. SISTEMA DE APRENDIZAJE AUTOMÁTICO ROBUSTO
# ======================
class HookOptimizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = None
        self.hooks_db = []

    def entrenar(self, hooks_virales):
        """Versión mejorada con manejo de errores."""
        if not hooks_virales or len(hooks_virales) < 2:
            st.warning("No hay suficientes hooks virales para entrenar el optimizador. Usando hooks por defecto.")
            return False

        try:
            X = self.vectorizer.fit_transform(hooks_virales)
            # Asegura que n_clusters no sea mayor que el número de muestras disponibles
            n_clusters = min(3, len(hooks_virales) - 1)
            if n_clusters < 1:
                n_clusters = 1
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.model.fit(X)
            self.hooks_db = hooks_virales
            return True
        except Exception as e:
            st.error(f"Error en entrenamiento del HookOptimizer: {str(e)}")
            return False

    def generar_hook_optimizado(self, texto, tema):
        """Genera hooks contextuales usando detección de entidades del script."""
        try:
            # Obtener entidades relevantes del script
            personas = extraer_entidades(texto, "PER")
            organizaciones = extraer_entidades(texto, "ORG")
            productos = extraer_entidades(texto, "PRODUCT")
            lugares = extraer_entidades(texto, "LOC")
            fechas = extraer_entidades(texto, "DATE")
            # Extraer nombres de animales específicos si están en el script
            nombres_animales_en_script = []
            for animal_nombre in ["perro", "gato", "hámster", "loro", "cachorros", "golden retriever", "labrador"]:
                if re.search(rf"\b{animal_nombre}\b", texto.lower()):
                    nombres_animales_en_script.append(animal_nombre)
            
            # Priorizar hooks de la temática detectada
            if tema in TEMATICAS:
                hooks_tema = TEMATICAS[tema]["hooks"]
                estrategia = random.choice(list(hooks_tema.keys()))
                plantilla = random.choice(hooks_tema[estrategia])

                hook = plantilla
                # Reemplazos para entidades
                if "{piloto}" in hook and personas: hook = hook.replace("{piloto}", random.choice(personas))
                if "{equipo}" in hook and organizaciones: hook = hook.replace("{equipo}", random.choice(organizaciones))
                if "{circuito}" in hook and lugares: hook = hook.replace("{circuito}", random.choice(lugares))
                if "{robot}" in hook and productos: hook = hook.replace("{robot}", random.choice(productos))
                elif "{robot}" in hook and personas: hook = hook.replace("{robot}", random.choice(personas))
                if "{compañía}" in hook and organizaciones: hook = hook.replace("{compañía}", random.choice(organizaciones))
                if "{evento}" in hook and fechas: hook = hook.replace("{evento}", random.choice(fechas))
                if "{jugador}" in hook and personas: hook = hook.replace("{jugador}", random.choice(personas))
                if "{producto}" in hook and productos: hook = hook.replace("{producto}", random.choice(productos))
                if "{persona}" in hook and personas: hook = hook.replace("{persona}", random.choice(personas))
                
                # *** AJUSTE CLAVE AQUÍ para {animal} y {tipo_mascota} ***
                if "{animal}" in hook:
                    if nombres_animales_en_script:
                        hook = hook.replace("{animal}", random.choice(nombres_animales_en_script))
                    elif personas: # Si hay nombres propios (ej. Firulais)
                        hook = hook.replace("{animal}", random.choice(personas))
                    else: # Fallback general
                        hook = hook.replace("{animal}", random.choice(["tu adorable mascota", "este peludo amigo", "este travieso animal"]))

                if "{tipo_mascota}" in hook:
                    if nombres_animales_en_script:
                        hook = hook.replace("{tipo_mascota}", random.choice(nombres_animales_en_script))
                    else: # Fallback general
                        hook = hook.replace("{tipo_mascota}", random.choice(["perro", "gato", "loro", "hámster"]))

                if "{mal_hábito}" in hook:
                    hook = hook.replace("{mal_hábito}", random.choice(["ladrar mucho", "arañar muebles", "morder cables"]))


                # Fallback para placeholders genéricos
                hook = hook.replace("{robot}", "Ameca") \
                           .replace("{tecnología}", "robótica") \
                           .replace("{industria}", "la interacción humano-máquina") \
                           .replace("{sistema}", "sistema avanzado") \
                           .replace("{evento}", "gran carrera") \
                           .replace("{marca}", "velocidad récord") \
                           .replace("{año}", str(datetime.now().year)) \
                           .replace("{jugador}", random.choice(["Messi", "Cristiano", "Haaland"])) \
                           .replace("{sistema_juego}", "el 4-4-2") \
                           .replace("{incidente}", "incidente polémico") \
                           .replace("{estadística}", "estadística asombrosa") \
                           .replace("{nombre_robot}", "Atlas de Boston Dynamics") \
                           .replace("{accion_mejorada}", "navegar con destreza") \
                           .replace("{algoritmo_ia}", "algoritmo de IA de vanguardia") \
                           .replace("{proceso_robotico}", "eficiencia en la producción") \
                           .replace("{productividad}", "productividad") \
                           .replace("{seguridad}", "seguridad") \
                           .replace("{sector}", "la manufactura") \
                           .replace("{tipo_empresa}", "PYMES") \
                           .replace("{próxima_decada}", "próxima década") \
                           .replace("{sistema_robotico}", "el sistema quirúrgico") \
                           .replace("{procedimiento_medico}", "cirugías complejas") \
                           .replace("{condicion_paciente}", "rehabilitación") \
                           .replace("{ambito_salud}", "centros de salud") \
                           .replace("{hábito}", "hábito diario") \
                           .replace("{métrica}", "tu rendimiento") \
                           .replace("{situacion}", "la nada") \
                           .replace("{logro}", "el éxito") \
                           .replace("{tiempo}", "poco tiempo") \
                           .replace("{cantidad}", "mucho dinero") \
                           .replace("{método}", "un método probado") \
                           .replace("{porcentaje}", "un alto porcentaje") \
                           .replace("{dato_impactante}", "un dato sorprendente") \
                           .replace("{novedad}", "la última novedad") \
                           .replace("{pregunta}", "¿Estás de acuerdo?") \
                           .replace("{tema}", tema)

                return hook
            
            # Fallback si el tema no está definido
            return "Descubre cómo esto cambiará tu perspectiva para siempre."
        except Exception as e:
            st.error(f"Error en generar_hook_optimizado: {str(e)}")
            return "¡Esto es algo que no te puedes perder!"
            
# ======================
# FUNCIONES AUXILIARES AVANZADAS
# ======================

# Variable global para el modelo de SpaCy
nlp = None 

@st.cache_resource
def get_spacy_model():
    return spacy.load("es_core_news_sm")

def extraer_entidades(texto, tipo_entidad=None):
    """Extrae entidades nombradas (personas, organizaciones, lugares, productos) de un texto usando SpaCy."""
    if nlp is None: 
        st.error("Error: Modelo de SpaCy no cargado. Contacta al soporte.")
        return []
    doc = nlp(texto)
    entidades = []
    for ent in doc.ents:
        if tipo_entidad is None or ent.label_ == tipo_entidad:
            entidades.append(ent.text)
    return list(set(entidades))

# ======================
# 3. FUNCIONES PRINCIPALES ACTUALIZADAS
# ======================
def analizar_tematica(texto):
    """Detección mejorada de temática"""
    scores = defaultdict(int)
    for tema, data in TEMATICAS.items():
        for palabra in data["palabras_clave"]:
            if re.search(rf"\b{palabra}\b", texto.lower()):
                scores[tema] += 1
    
    if not scores:
        return ("General", 0)
    
    mejor_tema, puntaje = max(scores.items(), key=lambda x: x[1])
    confianza = min(100, puntaje * 20)
    return (mejor_tema, confianza)

def mejorar_script(script, tema):
    """Mejora scripts para cualquier temática con técnicas virales"""
    segmentos_temporales = re.findall(r"(\(\d+-\d+\ssegundos\).*)", script)
    tiene_estructura = bool(segmentos_temporales)
    
    mejoras_por_tema = {
        "Robótica": {
            "hooks": ["{robot} ahora puede {acción}", "La revolución de {tecnología} en {año}"],
            "transiciones": ["SFX: Sonido futurista", "Corte rápido a detalle tecnológico"],
            "estadisticas": ["{porcentaje}% más rápido", "Capacidad de {función} mejorada"]
        },
        "Fútbol": {
            "hooks": ["El {técnica} que cambió el partido", "{jugador} rompió el récord"],
            "transiciones": ["SFX: Hinchada", "Slow motion clave"],
            "estadisticas": ["{goles} goles en {minutos}", "Pase con {porcentaje}% precisión"]
        },
        "Finanzas": {
            "hooks": ["Cómo ahorré {cantidad} en {tiempo}", "El error que cuesta {porcentaje}% anual"],
            "transiciones": ["Gráfico animado", "Zoom a cifras clave"],
            "estadisticas": ["Rentabilidad del {porcentaje}%", "Ahorro de {tiempo} horas"]
        },
        "Mascotas": {
            "hooks": ["Las travesuras de {animal} que te harán el día", "Tu {tipo_mascota} es más inteligente de lo que crees"],
            "transiciones": ["SFX: Sonido de risas", "Corte a cara de sorpresa", "Música divertida"],
            "estadisticas": ["{numero} segundos de pura diversión", "Destrucción en {cantidad} minutos"]
        }
    }
    
    reemplazos = {
        "{año}": str(datetime.now().year),
        "{robot}": "Ameca" if tema == "Robótica" else "este dispositivo",
        "{jugador}": random.choice(["Messi", "Cristiano", "Haaland"]) if tema == "Fútbol" else "el protagonista",
        "{tema}": tema,
        "{animal}": random.choice(["perro", "gato", "hámster"]) if tema == "Mascotas" else "animal", # Asegurar fallback genérico aquí también
        "{tipo_mascota}": random.choice(["perro", "gato", "loro"]) if tema == "Mascotas" else "mascota", # Asegurar fallback genérico aquí también
        "{numero}": str(random.randint(10, 60)),
        "{cantidad}": str(random.randint(1, 10))
    }
    
    plantillas_genericas = {
        "hook_inicial": [
            "¿Sabías que...? {dato_impactante}",
            "🚨 ALERTA: {novedad} está cambiando las reglas"
        ],
        "mejora_visual": [
            "💡 PRO TIP: Usa primeros planos cada 3 segundos",
            "🎬 TÉCNICA: Cambio de ángulo tras cada afirmación"
        ],
        "llamado_accion": [
            "👇 ¿Qué opinas? Comenta '{pregunta}'",
            "🔥 No te pierdas más contenido como este → @tu_canal"
        ]
    }

    script_final_mejorado = []

    if tiene_estructura:
        lineas = script.split('\n')
        
        for linea in lineas:
            script_final_mejorado.append(linea)
            
            if re.search(r"^\(\d+-\d+\ssegundos\)", linea):
                mejora_opciones = mejoras_por_tema.get(tema, {}).get("transiciones")
                if mejora_opciones:
                    mejora = random.choice(mejora_opciones)
                else:
                    mejora = random.choice(plantillas_genericas["mejora_visual"])
                
                for k, v in reemplazos.items():
                    mejora = mejora.replace(k, v)
                
                script_final_mejorado.append(f"✨ MEJORA: {mejora}")
                
    else:
        hook_gen = generar_hook(tema, reemplazos) # Asegurarse de que el hook generado aquí se usa
        llamado_accion_gen = random.choice(plantillas_genericas["llamado_accion"])
        
        script_final_mejorado.append(f"(0-5 segundos) 🎯 GANCHO INICIAL: {hook_gen}")
        script_final_mejorado.append("\n" + script.strip() + "\n")
        script_final_mejorado.append(f"(FINAL) 📲 LLAMADA A LA ACCIÓN: {llamado_accion_gen}")
        script_final_mejorado.append(f"✨ SUGERENCIA VISUAL: Considera añadir cortes rápidos y música dinámica.")


    script_final = '\n'.join(script_final_mejorado)
    
    hashtags = TEMATICAS.get(tema, {}).get("hashtags", ["#Viral", "#Trending"])
    script_final += f"\n\n🔖 HASHTAGS: {' '.join(hashtags[:3])}"
    
    return script_final

def generar_hook(tema, reemplazos):
    """Genera hooks temáticos dinámicos"""
    hooks_tema = TEMATICAS.get(tema, {}).get("hooks", {})
    hooks_genericos = {
        "impacto": ["Lo que nadie te dijo sobre {tema}"],
        "curiosidad": ["¿Por qué {tema} está revolucionando todo?"]
    }
    
    hooks_disponibles = []
    if hooks_tema:
        for estrategia in hooks_tema.values():
            hooks_disponibles.extend(estrategia)
    
    for hook_gen in hooks_genericos.values():
        hooks_disponibles.extend(hook_gen)
    
    hook = random.choice(hooks_disponibles) if hooks_disponibles else "Descubre esto que cambiará tu perspectiva"
    
    for k, v in reemplazos.items():
        hook = hook.replace(k, v)
    
    return hook

# ======================
# 4. INTERFAZ STREAMLIT OPTIMIZADA
# ======================

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
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
        "Esta tecnología robótica te sorprenderá"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎬 Script para Analizar")
        texto = st.text_area("Pega tu script completo:", height=300,
                             placeholder="Ej: (0-3 segundos) Video impactante...")
        
    with col2:
        if st.button("🚀 Optimizar Contenido"):
            if texto:
                with st.spinner("Analizando y mejorando..."):
                    tema, confianza = analizar_tematica(texto)
                    blob = TextBlob(texto) 
                    polaridad = blob.sentiment.polarity
                    
                    # Esta variable 'hook' es la que necesitamos asegurar que se muestre.
                    generated_hook = hook_ai.generar_hook_optimizado(texto, tema) 
                    
                    script_mejorado = mejorar_script(texto, tema)
                    hashtags_output = ' '.join(TEMATICAS.get(tema, {}).get("hashtags", ["#Viral"]))

                    st.subheader(f"🎯 Temática: {tema} (Confianza: {confianza}%)")
                    
                    # Usamos 'generated_hook' aquí
                    st.text_area("Hook Viral Recomendado:", value=generated_hook, height=100) 
                    
                    st.text_area("Script Optimizado:", value=script_mejorado, height=300)
                    
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

                        st.write(f"🔍 Hashtags recomendados: {hashtags_output}")
            else:
                st.warning("Por favor ingresa un script para analizar")

if __name__ == "__main__":
    main()
