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
        "palabras_clave": ["f1", "gran premio", "piloto", "carrera", "escudería", "circuito", "clasificación", "trompos", "silverstone", "ferrari", "mercedes", "red bull", "aston martin", "alpine", "alonso", "hamilton", "verstappen", "sainz", "leclerc"], # Añadidas más palabras clave
        "hooks": {
            "técnica": ["El {sistema} que hizo a {equipo_f1} ganar en {circuito_f1}"],
            "polémica": ["La decisión de la FIA que cambió el {evento_f1_generico}"],
            "récord": ["{piloto_f1} rompió el récord de {marca} en {año}"],
            "inesperado": ["¡El {evento_f1_inesperado} más caótico en la historia de {circuito_f1}!", "Los {numero} trompos más salvajes de {evento_f1_generico}"] # NUEVAS PLANTILLAS
        },
        "hashtags": ["#F1", "#Formula1", "#F1News", "#SilverstoneF1", "#MotorSport"] # Más hashtags
    },
    "Fútbol": {
        "palabras_clave": ["gol", "partido", "jugador", "liga", "champions", "equipo", "copa", "mundial", "messi", "ronaldo", "mbappe"],
        "hooks": {
            "táctica": ["El {sistema_juego} que hizo campeón a {equipo_futbol}"],
            "polémica": ["El {incidente} más injusto de la historia del fútbol"],
            "dato": ["{jugador_futbol} tiene este récord de {estadística_futbol}"]
        },
        "hashtags": ["#Fútbol", "#Champions", "#LaLiga", "#FPC"]
    },

    # Robótica
    "Robots Humanoides": {
        "palabras_clave": ["humanoide", "bípedo", "androide", "atlas", "asimo", "ameca", "engineered arts", "robot"],
        "hooks": {
            "técnica": ["Los desafíos de la **locomoción bípeda** en {nombre_robot}"],
            "aplicación": ["Cómo los humanoides están revolucionando la {industria_robotica}"],
            "avance": ["El nuevo sensor de {compañia_robotica} que permite a los humanoides {accion_mejorada_robotica}"]
        },
        "hashtags": ["#Humanoides", "#RobotsHumanoides", "#Bípedos", "#Ameca", "#FuturoAI"]
    },
    "Inteligencia Artificial en Robótica": {
        "palabras_clave": ["ia", "aprendizaje automático", "machine learning", "visión artificial", "deep learning", "algoritmos", "inteligencia artificial"],
        "hooks": {
            "técnica": ["La **red neuronal** que permite a {robot_ia} reconocer {objeto_ia}"],
            "aplicación": ["IA para la **navegación autónoma** en {entorno_complejo}"],
            "impacto": ["Cómo el {algoritmo_ia} está optimizando el {proceso_robotico}"]
        },
        "hashtags": ["#AIRobótica", "#IA", "#MachineLearningRobots", "#VisiónArtificial", "#DeepLearning"]
    },
    "Robots Colaborativos (Cobots)": {
        "palabras_clave": ["cobots", "colaborativos", "seguridad", "interacción h-r", "industria 4.0", "manufactura"],
        "hooks": {
            "beneficio": ["**Cobots**: Mejorando la {productividad} y la {seguridad} en {sector_industrial}"],
            "implementación": ["Desafíos y soluciones al integrar **cobots** en {tipo_empresa}"],
            "futuro": ["El rol de los **robots colaborativos** en la {proxima_decada}"]
        },
        "hashtags": ["#Cobots", "#RobotsColaborativos", "#Industria40", "#Automatización"]
    },
    "Robótica Médica": {
        "palabras_clave": ["cirugía robótica", "quirúrgico", "da vinci", "rehabilitación", "exosqueletos", "telemedicina", "salud", "hospital"],
        "hooks": {
            "innovación": ["**Robótica médica**: La precisión de {sistema_robotico_medico} en {procedimiento_medico}"],
            "impacto_paciente": ["Cómo los **exosqueletos** están transformando la {condicion_paciente}"],
            "futuro": ["La próxima generación de **robots asistenciales** en {ambito_salud}"]
        },
        "hashtags": ["#RobóticaMédica", "#CirugíaRobótica", "#Exoesqueletos", "#SaludDigital", "#Medicina"]
    },

    # Mascotas
    "Mascotas": {
        "palabras_clave": ["perro", "gato", "hámster", "pájaro", "mascota", "animales", "cachorros", "golden retriever", "labrador", "loro", "periquito", "conejo", "hurón"],
        "hooks": {
            "humor": ["Tu mascota también hace ESTO para volverte loco", "¿Listo para reírte? Las travesuras más épicas de {animal_mascota}"],
            "consejo": ["El secreto para que tu {tipo_mascota} deje de {mal_habito_mascota}"],
            "emocional": ["La historia de {animal_mascota} que te derretirá el corazón"]
        },
        "hashtags": ["#Mascotas", "#AnimalesGraciosos", "#MascotasVirales", "#PetsOfTikTok", "#AmorAnimal"]
    },

    # Mindset
    "Mindset": {
        "palabras_clave": ["éxito", "hábitos", "mentalidad", "crecimiento", "productividad", "motivación", "superación"],
        "hooks": {
            "científico": ["Estudio de Harvard prueba que {hábito_mindset} aumenta {metrica_mindset}"],
            "inspiración": ["Cómo {persona_mindset} pasó de {situacion_mindset} a {logro_mindset}"],
            "acción": ["Si haces esto cada mañana, tu vida cambiará en {tiempo_mindset}"]
        },
        "hashtags": ["#Mindset", "#CrecimientoPersonal", "#Motivacion", "#Productividad"]
    },

    # Finanzas
    "Finanzas": {
        "palabras_clave": ["dinero", "inversión", "ahorro", "finanzas", "criptomonedas", "bolsa", "negocios", "emprendimiento"],
        "hooks": {
            "impacto": ["Cómo ahorré {cantidad_dinero} en {tiempo_finanzas} con {metodo_finanzas}"],
            "error": ["El error que te hace perder {porcentaje_finanzas}% de tus ingresos"],
            "sistema": ["El método {nombre_metodo} para multiplicar tu dinero"]
        },
        "hashtags": ["#Finanzas", "#Ahorro", "#Inversión", "#Dinero", "#Emprendimiento"]
    },

    # Tecnología (General)
    "Tecnología": {
        "palabras_clave": ["robot", "ia", "tecnología", "automatización", "innovación", "gadget", "futuro", "ciencia"],
        "hooks": {
            "futuro": ["Cómo {tecnologia_general} cambiará {industria_general} en {año}"],
            "comparación": ["{ProductoA_tech} vs {ProductoB_tech}: ¿Cuál gana?"],
            "review": ["Probé {producto_tech_review} y esto pasó"]
        },
        "hashtags": ["#Tecnología", "#Innovación", "#Gadgets", "#Ciencia", "#Futuro"]
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
            # Extraer entidades usando SpaCy
            personas = extraer_entidades(texto, "PER")
            organizaciones = extraer_entidades(texto, "ORG")
            productos = extraer_entidades(texto, "PRODUCT")
            lugares = extraer_entidades(texto, "LOC")
            fechas = extraer_entidades(texto, "DATE")
            
            # Entidades específicas de contexto para F1, Mascotas, etc.
            f1_equipos = [org for org in organizaciones if org.lower() in ["ferrari", "mercedes", "red bull", "aston martin", "alpine"]]
            f1_pilotos = [per for per in personas if per.lower() in ["alonso", "hamilton", "verstappen", "sainz", "leclerc", "perez", "russell"]]
            f1_circuitos = [loc for loc in lugares if loc.lower() in ["silverstone", "monza", "spa", "montmelo", "monaco"]]

            nombres_animales_en_script = []
            for animal_nombre in ["perro", "gato", "hámster", "loro", "cachorros", "golden retriever", "labrador", "ave", "conejo", "hurón"]:
                if re.search(rf"\b{animal_nombre}\b", texto.lower()):
                    nombres_animales_en_script.append(animal_nombre)
            
            if tema in TEMATICAS:
                hooks_tema = TEMATICAS[tema]["hooks"]
                estrategia = random.choice(list(hooks_tema.keys()))
                plantilla = random.choice(hooks_tema[estrategia])

                hook = plantilla
                
                # REEMPLAZOS ESPECÍFICOS POR TEMÁTICA
                if tema == "Fórmula 1":
                    if "{piloto_f1}" in hook: hook = hook.replace("{piloto_f1}", random.choice(f1_pilotos) if f1_pilotos else random.choice(["Verstappen", "Hamilton"]))
                    if "{equipo_f1}" in hook: hook = hook.replace("{equipo_f1}", random.choice(f1_equipos) if f1_equipos else random.choice(["Red Bull", "Ferrari"]))
                    if "{circuito_f1}" in hook: hook = hook.replace("{circuito_f1}", random.choice(f1_circuitos) if f1_circuitos else random.choice(["Silverstone", "Mónaco"]))
                    if "{evento_f1_generico}" in hook: hook = hook.replace("{evento_f1_generico}", random.choice(["Gran Premio", "clasificación", "carrera"]))
                    if "{evento_f1_inesperado}" in hook: hook = hook.replace("{evento_f1_inesperado}", random.choice(["Gran Premio", "sesión de clasificación"]))
                    if "{numero}" in hook: hook = hook.replace("{numero}", str(random.randint(3, 10))) # Para "Los X trompos"

                elif tema == "Mascotas":
                    if "{animal_mascota}" in hook:
                        if nombres_animales_en_script:
                            hook = hook.replace("{animal_mascota}", random.choice(nombres_animales_en_script))
                        elif personas: # Si hay nombres propios (ej. Firulais)
                            hook = hook.replace("{animal_mascota}", random.choice(personas))
                        else: # Fallback general
                            hook = hook.replace("{animal_mascota}", random.choice(["tu adorable mascota", "este peludo amigo", "este travieso animal"]))

                    if "{tipo_mascota}" in hook:
                        if nombres_animales_en_script:
                            hook = hook.replace("{tipo_mascota}", random.choice(nombres_animales_en_script))
                        else: # Fallback general
                            hook = hook.replace("{tipo_mascota}", random.choice(["perro", "gato", "loro", "hámster"]))

                    if "{mal_habito_mascota}" in hook:
                        hook = hook.replace("{mal_habito_mascota}", random.choice(["ladrar mucho", "arañar muebles", "morder cables", "comerse los zapatos"]))

                # REEMPLAZOS GENÉRICOS (para placeholders que pueden aparecer en varias temáticas)
                if "{año}" in hook: hook = hook.replace("{año}", str(datetime.now().year))
                if "{marca}" in hook: hook = hook.replace("{marca}", "velocidad récord")
                if "{incidente}" in hook: hook = hook.replace("{incidente}", "incidente polémico")
                if "{dato_impactante}" in hook: hook = hook.replace("{dato_impactante}", "un dato sorprendente")
                if "{novedad}" in hook: hook = hook.replace("{novedad}", "la última novedad")
                if "{pregunta}" in hook: hook = hook.replace("{pregunta}", "¿Estás de acuerdo?")
                if "{tema}" in hook: hook = hook.replace("{tema}", tema)

                # Robótica
                if "{nombre_robot}" in hook: hook = hook.replace("{nombre_robot}", random.choice(productos) if productos else "Ameca")
                if "{industria_robotica}" in hook: hook = hook.replace("{industria_robotica}", "la manufactura")
                if "{compañia_robotica}" in hook: hook = hook.replace("{compañia_robotica}", random.choice(organizaciones) if organizaciones else "Engineered Arts")
                if "{accion_mejorada_robotica}" in hook: hook = hook.replace("{accion_mejorada_robotica}", "navegar con destreza")
                if "{robot_ia}" in hook: hook = hook.replace("{robot_ia}", random.choice(productos) if productos else "un robot de IA")
                if "{objeto_ia}" in hook: hook = hook.replace("{objeto_ia}", "objetos complejos")
                if "{entorno_complejo}" in hook: hook = hook.replace("{entorno_complejo}", "ciudades inteligentes")
                if "{algoritmo_ia}" in hook: hook = hook.replace("{algoritmo_ia}", "un algoritmo de IA de vanguardia")
                if "{proceso_robotico}" in hook: hook = hook.replace("{proceso_robotico}", "eficiencia en la producción")
                if "{productividad}" in hook: hook = hook.replace("{productividad}", "productividad")
                if "{seguridad}" in hook: hook = hook.replace("{seguridad}", "seguridad")
                if "{sector_industrial}" in hook: hook = hook.replace("{sector_industrial}", "la manufactura")
                if "{tipo_empresa}" in hook: hook = hook.replace("{tipo_empresa}", "PYMES")
                if "{proxima_decada}" in hook: hook = hook.replace("{proxima_decada}", "próxima década")
                if "{sistema_robotico_medico}" in hook: hook = hook.replace("{sistema_robotico_medico}", "el sistema quirúrgico Da Vinci")
                if "{procedimiento_medico}" in hook: hook = hook.replace("{procedimiento_medico}", "cirugías complejas")
                if "{condicion_paciente}" in hook: hook = hook.replace("{condicion_paciente}", "rehabilitación")
                if "{ambito_salud}" in hook: hook = hook.replace("{ambito_salud}", "centros de salud")

                # Mindset
                if "{hábito_mindset}" in hook: hook = hook.replace("{hábito_mindset}", random.choice(["leer a diario", "meditar", "hacer ejercicio"]))
                if "{metrica_mindset}" in hook: hook = hook.replace("{metrica_mindset}", "tu creatividad")
                if "{persona_mindset}" in hook: hook = hook.replace("{persona_mindset}", random.choice(personas) if personas else "una persona promedio")
                if "{situacion_mindset}" in hook: hook = hook.replace("{situacion_mindset}", "la nada")
                if "{logro_mindset}" in hook: hook = hook.replace("{logro_mindset}", "el éxito")
                if "{tiempo_mindset}" in hook: hook = hook.replace("{tiempo_mindset}", "poco tiempo")

                # Finanzas
                if "{cantidad_dinero}" in hook: hook = hook.replace("{cantidad_dinero}", "1000 dólares")
                if "{tiempo_finanzas}" in hook: hook = hook.replace("{tiempo_finanzas}", "3 meses")
                if "{metodo_finanzas}" in hook: hook = hook.replace("{metodo_finanzas}", "un método probado")
                if "{porcentaje_finanzas}" in hook: hook = hook.replace("{porcentaje_finanzas}", "20")
                if "{nombre_metodo}" in hook: hook = hook.replace("{nombre_metodo}", "Warren Buffett")

                # Tecnología General
                if "{tecnologia_general}" in hook: hook = hook.replace("{tecnologia_general}", "la realidad virtual")
                if "{industria_general}" in hook: hook = hook.replace("{industria_general}", "la educación")
                if "{ProductoA_tech}" in hook: hook = hook.replace("{ProductoA_tech}", "iPhone 16")
                if "{ProductoB_tech}" in hook: hook = hook.replace("{ProductoB_tech}", "Galaxy S25")
                if "{producto_tech_review}" in hook: hook = hook.replace("{producto_tech_review}", random.choice(productos) if productos else "el nuevo smartphone")

                return hook
            
            # Fallback si el tema no está definido
            return "Descubre cómo esto cambiará tu perspectiva para siempre."
        except Exception as e:
            st.error(f"Error en generar_hook_optimizado: {str(e)}")
            return "¡Esto es algo que no te puedes perder!"
            
# ======================
# FUNCIONES AUXILIARES AVANZADAS
# ======================

nlp = None 

@st.cache_resource
def get_spacy_model():
    """Carga el modelo de SpaCy para español."""
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        st.error("Modelo 'es_core_news_sm' de SpaCy no encontrado. Intentando descargar...")
        spacy.cli.download("es_core_news_sm")
        return spacy.load("es_core_news_sm")

def extraer_entidades(texto, tipo_entidad=None):
    """Extrae entidades nombradas (personas, organizaciones, lugares, productos) de un texto usando SpaCy."""
    if nlp is None: 
        # Esto no debería pasar si get_spacy_model se llamó correctamente en main
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
    """Detección mejorada de temática con mayor confianza por palabra clave."""
    scores = defaultdict(int)
    for tema, data in TEMATICAS.items():
        for palabra in data["palabras_clave"]:
            # Usamos re.search para encontrar la palabra completa, no solo substrings
            # y añadimos peso a las palabras clave para mayor precisión
            if re.search(rf"\b{palabra}\b", texto.lower()):
                scores[tema] += 1
    
    if not scores:
        return ("General", 0)
    
    mejor_tema, puntaje = max(scores.items(), key=lambda x: x[1])
    confianza = min(100, puntaje * 20) # Escala a porcentaje, 5 palabras clave = 100%
    return (mejor_tema, confianza)

def mejorar_script(script, tema, pre_generated_hook=None):
    """Mejora scripts para cualquier temática con técnicas virales."""
    # Detectar si el script ya tiene marcas de tiempo explícitas
    # Se ajustó el regex para ser más flexible con 's' o 'segundos' y ser case-insensitive
    segmentos_temporales = re.findall(r"(\(\d+-\d+\s*(?:segundos|s)\).*)", script, re.IGNORECASE)
    tiene_estructura = bool(segmentos_temporales)
    
    mejoras_por_tema = {
        "Robótica": {
            "transiciones": ["SFX: Sonido futurista activándose", "Corte rápido a detalle de mecanismo", "✨ MEJORA: Toma de Ameca expresando una emoción sutil"],
        },
        "Fútbol": {
            "transiciones": ["SFX: Hinchada rugiendo", "Slow motion de jugada clave", "✨ MEJORA: Gráfico animado de estadística de jugador"],
        },
        "Finanzas": {
            "transiciones": ["Gráfico animado de crecimiento/caída", "Zoom a cifras clave", "SFX: Sonido de calculadora o transacción"],
        },
        "Mascotas": {
            "transiciones": ["SFX: Sonido de risas o asombro", "Corte a cara de sorpresa del dueño", "Música divertida subiendo", "✨ MEJORA: Primer plano a la expresión traviesa de la mascota"],
        },
        "Fórmula 1": {
            "transiciones": ["SFX: Chirrido de neumáticos", "Cámara lenta del trompo", "✨ MEJORA: Toma en cabina del piloto reaccionando", "Corte rápido entre diferentes ángulos de la acción"],
        }
    }
    
    # Reemplazos genéricos para mejoras y CTA
    reemplazos_genericos = {
        "{numero}": str(random.randint(10, 60)),
        "{cantidad}": str(random.randint(1, 10)),
        "{pregunta}": "¿Qué te pareció?", # para la CTA genérica
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
        
        # Iterar sobre las líneas para insertar mejoras después de cada marca de tiempo
        for i, linea in enumerate(lineas):
            script_final_mejorado.append(linea)
            
            # Si la línea contiene una marca de tiempo
            if re.search(r"^\(\d+-\d+\s*(?:segundos|s)\)", linea, re.IGNORECASE):
                mejora_opciones = mejoras_por_tema.get(tema, {}).get("transiciones")
                if mejora_opciones:
                    mejora = random.choice(mejora_opciones)
                else: # Fallback a mejoras visuales genéricas
                    mejora = random.choice(plantillas_genericas["mejora_visual"])
                
                # Aplicar reemplazos genéricos a la mejora
                for k, v in reemplazos_genericos.items():
                    mejora = mejora.replace(k, v)
                
                # Añadir la mejora
                script_final_mejorado.append(f"✨ MEJORA: {mejora}")
                
        # Al final del script con estructura, añadir un CTA si no se incluyó ya
        # Se asume que el script original tiene un CTA, pero si no, se añade uno genérico
        # Se busca si ya existe un CTA evidente en las últimas líneas
        cta_ya_presente = any(re.search(r"(comenta|suscribe|siguenos)", l.lower()) for l in lineas[-5:])

        if not cta_ya_presente:
            llamado_accion_gen = random.choice(plantillas_genericas["llamado_accion"])
            for k, v in reemplazos_genericos.items():
                llamado_accion_gen = llamado_accion_gen.replace(k, v)
            script_final_mejorado.append(f"\n(FINAL) 📲 LLAMADA A LA ACCIÓN: {llamado_accion_gen}")
            script_final_mejorado.append(f"✨ SUGERENCIA VISUAL: Considera añadir cortes rápidos y música dinámica.")
            
    else: # Si el script NO tiene una estructura temporal explícita
        # Usamos el hook pre-generado si existe, de lo contrario generamos uno genérico
        gancho_a_usar = pre_generated_hook if pre_generated_hook else generar_hook(tema, reemplazos_genericos)
        llamado_accion_gen = random.choice(plantillas_genericas["llamado_accion"])
        for k, v in reemplazos_genericos.items():
            llamado_accion_gen = llamado_accion_gen.replace(k, v)
        
        script_final_mejorado.append(f"(0-5 segundos) 🎯 GANCHO INICIAL: {gancho_a_usar}")
        script_final_mejorado.append("\n" + script.strip() + "\n") # Insertar el script original completo
        script_final_mejorado.append(f"(FINAL) 📲 LLAMADA A LA ACCIÓN: {llamado_accion_gen}")
        script_final_mejorado.append(f"✨ SUGERENCIA VISUAL: Considera añadir cortes rápidos y música dinámica.")

    script_final = '\n'.join(script_final_mejorado)
    
    # Añadir hashtags al final
    hashtags = TEMATICAS.get(tema, {}).get("hashtags", ["#Viral", "#Trending"])
    script_final += f"\n\n🔖 HASHTAGS: {' '.join(hashtags[:4])}" # Limitar a 4 hashtags para mayor impacto
    
    return script_final

def generar_hook(tema, reemplazos):
    """Genera hooks temáticos dinámicos (usado como fallback o para hooks genéricos)."""
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
    
    # Aplicar reemplazos generales a los hooks generados aquí
    for k, v in reemplazos.items():
        hook = hook.replace(k, v)
    
    return hook

# ======================
# 4. INTERFAZ STREAMLIT OPTIMIZADA
# ======================

@st.cache_resource
def download_nltk_data():
    """Descarga los recursos de NLTK necesarios."""
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
    nlp = get_spacy_model() # Cargar el modelo de SpaCy al inicio

    # Entrenar el optimizador de hooks con ejemplos relevantes
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
        "El error de Hamilton que le costó la carrera"
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
                    
                    # Generamos el hook principal aquí
                    generated_hook = hook_ai.generar_hook_optimizado(texto, tema) 
                    
                    # Pasamos el hook principal a mejorar_script
                    script_mejorado = mejorar_script(texto, tema, pre_generated_hook=generated_hook)
                    
                    # Los hashtags ahora se generan dentro de mejorar_script, solo se muestran aquí
                    # Si quieres una lista separada de hashtags, deberías extraerla del final de script_mejorado
                    # o pasarla como un return adicional desde mejorar_script.
                    # Por ahora, simplemente tomaremos los hashtags de la temática para mostrar.
                    hashtags_display = ' '.join(TEMATICAS.get(tema, {}).get("hashtags", ["#Viral"]))

                    st.subheader(f"🎯 Temática: **{tema}** (Confianza: **{confianza}%**)")
                    
                    st.text_area("Hook Viral Recomendado:", value=generated_hook, height=100) 
                    
                    st.text_area("Script Optimizado:", value=script_mejorado, height=450) # Aumentado el tamaño
                    
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
    main()import streamlit as st
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
        "palabras_clave": ["f1", "gran premio", "piloto", "carrera", "escudería", "circuito", "clasificación", "trompos", "silverstone", "ferrari", "mercedes", "red bull", "aston martin", "alpine", "alonso", "hamilton", "verstappen", "sainz", "leclerc"], # Añadidas más palabras clave
        "hooks": {
            "técnica": ["El {sistema} que hizo a {equipo_f1} ganar en {circuito_f1}"],
            "polémica": ["La decisión de la FIA que cambió el {evento_f1_generico}"],
            "récord": ["{piloto_f1} rompió el récord de {marca} en {año}"],
            "inesperado": ["¡El {evento_f1_inesperado} más caótico en la historia de {circuito_f1}!", "Los {numero} trompos más salvajes de {evento_f1_generico}"] # NUEVAS PLANTILLAS
        },
        "hashtags": ["#F1", "#Formula1", "#F1News", "#SilverstoneF1", "#MotorSport"] # Más hashtags
    },
    "Fútbol": {
        "palabras_clave": ["gol", "partido", "jugador", "liga", "champions", "equipo", "copa", "mundial", "messi", "ronaldo", "mbappe"],
        "hooks": {
            "táctica": ["El {sistema_juego} que hizo campeón a {equipo_futbol}"],
            "polémica": ["El {incidente} más injusto de la historia del fútbol"],
            "dato": ["{jugador_futbol} tiene este récord de {estadística_futbol}"]
        },
        "hashtags": ["#Fútbol", "#Champions", "#LaLiga", "#FPC"]
    },

    # Robótica
    "Robots Humanoides": {
        "palabras_clave": ["humanoide", "bípedo", "androide", "atlas", "asimo", "ameca", "engineered arts", "robot"],
        "hooks": {
            "técnica": ["Los desafíos de la **locomoción bípeda** en {nombre_robot}"],
            "aplicación": ["Cómo los humanoides están revolucionando la {industria_robotica}"],
            "avance": ["El nuevo sensor de {compañia_robotica} que permite a los humanoides {accion_mejorada_robotica}"]
        },
        "hashtags": ["#Humanoides", "#RobotsHumanoides", "#Bípedos", "#Ameca", "#FuturoAI"]
    },
    "Inteligencia Artificial en Robótica": {
        "palabras_clave": ["ia", "aprendizaje automático", "machine learning", "visión artificial", "deep learning", "algoritmos", "inteligencia artificial"],
        "hooks": {
            "técnica": ["La **red neuronal** que permite a {robot_ia} reconocer {objeto_ia}"],
            "aplicación": ["IA para la **navegación autónoma** en {entorno_complejo}"],
            "impacto": ["Cómo el {algoritmo_ia} está optimizando el {proceso_robotico}"]
        },
        "hashtags": ["#AIRobótica", "#IA", "#MachineLearningRobots", "#VisiónArtificial", "#DeepLearning"]
    },
    "Robots Colaborativos (Cobots)": {
        "palabras_clave": ["cobots", "colaborativos", "seguridad", "interacción h-r", "industria 4.0", "manufactura"],
        "hooks": {
            "beneficio": ["**Cobots**: Mejorando la {productividad} y la {seguridad} en {sector_industrial}"],
            "implementación": ["Desafíos y soluciones al integrar **cobots** en {tipo_empresa}"],
            "futuro": ["El rol de los **robots colaborativos** en la {proxima_decada}"]
        },
        "hashtags": ["#Cobots", "#RobotsColaborativos", "#Industria40", "#Automatización"]
    },
    "Robótica Médica": {
        "palabras_clave": ["cirugía robótica", "quirúrgico", "da vinci", "rehabilitación", "exosqueletos", "telemedicina", "salud", "hospital"],
        "hooks": {
            "innovación": ["**Robótica médica**: La precisión de {sistema_robotico_medico} en {procedimiento_medico}"],
            "impacto_paciente": ["Cómo los **exosqueletos** están transformando la {condicion_paciente}"],
            "futuro": ["La próxima generación de **robots asistenciales** en {ambito_salud}"]
        },
        "hashtags": ["#RobóticaMédica", "#CirugíaRobótica", "#Exoesqueletos", "#SaludDigital", "#Medicina"]
    },

    # Mascotas
    "Mascotas": {
        "palabras_clave": ["perro", "gato", "hámster", "pájaro", "mascota", "animales", "cachorros", "golden retriever", "labrador", "loro", "periquito", "conejo", "hurón"],
        "hooks": {
            "humor": ["Tu mascota también hace ESTO para volverte loco", "¿Listo para reírte? Las travesuras más épicas de {animal_mascota}"],
            "consejo": ["El secreto para que tu {tipo_mascota} deje de {mal_habito_mascota}"],
            "emocional": ["La historia de {animal_mascota} que te derretirá el corazón"]
        },
        "hashtags": ["#Mascotas", "#AnimalesGraciosos", "#MascotasVirales", "#PetsOfTikTok", "#AmorAnimal"]
    },

    # Mindset
    "Mindset": {
        "palabras_clave": ["éxito", "hábitos", "mentalidad", "crecimiento", "productividad", "motivación", "superación"],
        "hooks": {
            "científico": ["Estudio de Harvard prueba que {hábito_mindset} aumenta {metrica_mindset}"],
            "inspiración": ["Cómo {persona_mindset} pasó de {situacion_mindset} a {logro_mindset}"],
            "acción": ["Si haces esto cada mañana, tu vida cambiará en {tiempo_mindset}"]
        },
        "hashtags": ["#Mindset", "#CrecimientoPersonal", "#Motivacion", "#Productividad"]
    },

    # Finanzas
    "Finanzas": {
        "palabras_clave": ["dinero", "inversión", "ahorro", "finanzas", "criptomonedas", "bolsa", "negocios", "emprendimiento"],
        "hooks": {
            "impacto": ["Cómo ahorré {cantidad_dinero} en {tiempo_finanzas} con {metodo_finanzas}"],
            "error": ["El error que te hace perder {porcentaje_finanzas}% de tus ingresos"],
            "sistema": ["El método {nombre_metodo} para multiplicar tu dinero"]
        },
        "hashtags": ["#Finanzas", "#Ahorro", "#Inversión", "#Dinero", "#Emprendimiento"]
    },

    # Tecnología (General)
    "Tecnología": {
        "palabras_clave": ["robot", "ia", "tecnología", "automatización", "innovación", "gadget", "futuro", "ciencia"],
        "hooks": {
            "futuro": ["Cómo {tecnologia_general} cambiará {industria_general} en {año}"],
            "comparación": ["{ProductoA_tech} vs {ProductoB_tech}: ¿Cuál gana?"],
            "review": ["Probé {producto_tech_review} y esto pasó"]
        },
        "hashtags": ["#Tecnología", "#Innovación", "#Gadgets", "#Ciencia", "#Futuro"]
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
            # Extraer entidades usando SpaCy
            personas = extraer_entidades(texto, "PER")
            organizaciones = extraer_entidades(texto, "ORG")
            productos = extraer_entidades(texto, "PRODUCT")
            lugares = extraer_entidades(texto, "LOC")
            fechas = extraer_entidades(texto, "DATE")
            
            # Entidades específicas de contexto para F1, Mascotas, etc.
            f1_equipos = [org for org in organizaciones if org.lower() in ["ferrari", "mercedes", "red bull", "aston martin", "alpine"]]
            f1_pilotos = [per for per in personas if per.lower() in ["alonso", "hamilton", "verstappen", "sainz", "leclerc", "perez", "russell"]]
            f1_circuitos = [loc for loc in lugares if loc.lower() in ["silverstone", "monza", "spa", "montmelo", "monaco"]]

            nombres_animales_en_script = []
            for animal_nombre in ["perro", "gato", "hámster", "loro", "cachorros", "golden retriever", "labrador", "ave", "conejo", "hurón"]:
                if re.search(rf"\b{animal_nombre}\b", texto.lower()):
                    nombres_animales_en_script.append(animal_nombre)
            
            if tema in TEMATICAS:
                hooks_tema = TEMATICAS[tema]["hooks"]
                
                # Prioritize 'inesperado' strategy if relevant keywords are found for F1
                # or 'humor' for pets if the script is about funny animal behavior
                estrategia = random.choice(list(hooks_tema.keys())) # Default random choice
                
                if tema == "Fórmula 1":
                    if re.search(r'\b(trompos|spin|accidente|caos|inesperado)\b', texto.lower()) and "inesperado" in hooks_tema:
                        estrategia = "inesperado"
                elif tema == "Mascotas":
                    if re.search(r'\b(chistoso|gracioso|divertido|travesuras|humor)\b', texto.lower()) and "humor" in hooks_tema:
                        estrategia = "humor"
                
                plantilla = random.choice(hooks_tema[estrategia])

                hook = plantilla
                
                # REEMPLAZOS ESPECÍFICOS POR TEMÁTICA
                if tema == "Fórmula 1":
                    if "{piloto_f1}" in hook: hook = hook.replace("{piloto_f1}", random.choice(f1_pilotos) if f1_pilotos else random.choice(["Verstappen", "Hamilton"]))
                    if "{equipo_f1}" in hook: hook = hook.replace("{equipo_f1}", random.choice(f1_equipos) if f1_equipos else random.choice(["Red Bull", "Ferrari"]))
                    if "{circuito_f1}" in hook: hook = hook.replace("{circuito_f1}", random.choice(f1_circuitos) if f1_circuitos else random.choice(["Silverstone", "Mónaco"]))
                    if "{evento_f1_generico}" in hook: hook = hook.replace("{evento_f1_generico}", random.choice(["Gran Premio", "clasificación", "carrera"]))
                    if "{evento_f1_inesperado}" in hook: hook = hook.replace("{evento_f1_inesperado}", random.choice(["Gran Premio", "sesión de clasificación"]))
                    if "{numero}" in hook: hook = hook.replace("{numero}", str(random.randint(3, 10))) # For "Los X trompos"

                elif tema == "Mascotas":
                    if "{animal_mascota}" in hook:
                        if nombres_animales_en_script:
                            hook = hook.replace("{animal_mascota}", random.choice(nombres_animales_en_script))
                        elif personas: # If there are proper names (e.g. Firulais)
                            hook = hook.replace("{animal_mascota}", random.choice(personas))
                        else: # General fallback
                            hook = hook.replace("{animal_mascota}", random.choice(["tu adorable mascota", "este peludo amigo", "este travieso animal"]))

                    if "{tipo_mascota}" in hook:
                        if nombres_animales_en_script:
                            hook = hook.replace("{tipo_mascota}", random.choice(nombres_animales_en_script))
                        else: # General fallback
                            hook = hook.replace("{tipo_mascota}", random.choice(["perro", "gato", "loro", "hámster"]))

                    if "{mal_habito_mascota}" in hook:
                        hook = hook.replace("{mal_habito_mascota}", random.choice(["ladrar mucho", "arañar muebles", "morder cables", "comerse los zapatos"]))

                # REEMPLAZOS GENÉRICOS (for placeholders that can appear in multiple themes)
                if "{año}" in hook: hook = hook.replace("{año}", str(datetime.now().year))
                if "{marca}" in hook: hook = hook.replace("{marca}", "velocidad récord")
                if "{incidente}" in hook: hook = hook.replace("{incidente}", "incidente polémico")
                if "{dato_impactante}" in hook: hook = hook.replace("{dato_impactante}", "un dato sorprendente")
                if "{novedad}" in hook: hook = hook.replace("{novedad}", "la última novedad")
                if "{pregunta}" in hook: hook = hook.replace("{pregunta}", "¿Estás de acuerdo?")
                if "{tema}" in hook: hook = hook.replace("{tema}", tema)
                if "{sistema}" in hook: hook = hook.replace("{sistema}", "sistema secreto") # Fallback for {system}

                # Robotics
                if "{nombre_robot}" in hook: hook = hook.replace("{nombre_robot}", random.choice(productos) if productos else "Ameca")
                if "{industria_robotica}" in hook: hook = hook.replace("{industria_robotica}", "la manufactura")
                if "{compañia_robotica}" in hook: hook = hook.replace("{compañia_robotica}", random.choice(organizaciones) if organizaciones else "Engineered Arts")
                if "{accion_mejorada_robotica}" in hook: hook = hook.replace("{accion_mejorada_robotica}", "navegar con destreza")
                if "{robot_ia}" in hook: hook = hook.replace("{robot_ia}", random.choice(productos) if productos else "un robot de IA")
                if "{objeto_ia}" in hook: hook = hook.replace("{objeto_ia}", "objetos complejos")
                if "{entorno_complejo}" in hook: hook = hook.replace("{entorno_complejo}", "ciudades inteligentes")
                if "{algoritmo_ia}" in hook: hook = hook.replace("{algoritmo_ia}", "un algoritmo de IA de vanguardia")
                if "{proceso_robotico}" in hook: hook = hook.replace("{proceso_robotico}", "eficiencia en la producción")
                if "{productividad}" in hook: hook = hook.replace("{productividad}", "productividad")
                if "{seguridad}" in hook: hook = hook.replace("{seguridad}", "seguridad")
                if "{sector_industrial}" in hook: hook = hook.replace("{sector_industrial}", "la manufactura")
                if "{tipo_empresa}" in hook: hook = hook.replace("{tipo_empresa}", "PYMES")
                if "{proxima_decada}" in hook: hook = hook.replace("{proxima_decada}", "próxima década")
                if "{sistema_robotico_medico}" in hook: hook = hook.replace("{sistema_robotico_medico}", "el sistema quirúrgico Da Vinci")
                if "{procedimiento_medico}" in hook: hook = hook.replace("{procedimiento_medico}", "cirugías complejas")
                if "{condicion_paciente}" in hook: hook = hook.replace("{condicion_paciente}", "rehabilitación")
                if "{ambito_salud}" in hook: hook = hook.replace("{ambito_salud}", "centros de salud")

                # Mindset
                if "{hábito_mindset}" in hook: hook = hook.replace("{hábito_mindset}", random.choice(["leer a diario", "meditar", "hacer ejercicio"]))
                if "{metrica_mindset}" in hook: hook = hook.replace("{metrica_mindset}", "tu creatividad")
                if "{persona_mindset}" in hook: hook = hook.replace("{persona_mindset}", random.choice(personas) if personas else "una persona promedio")
                if "{situacion_mindset}" in hook: hook = hook.replace("{situacion_mindset}", "la nada")
                if "{logro_mindset}" in hook: hook = hook.replace("{logro_mindset}", "el éxito")
                if "{tiempo_mindset}" in hook: hook = hook.replace("{tiempo_mindset}", "poco tiempo")

                # Finances
                if "{cantidad_dinero}" in hook: hook = hook.replace("{cantidad_dinero}", "1000 dólares")
                if "{tiempo_finanzas}" in hook: hook = hook.replace("{tiempo_finanzas}", "3 meses")
                if "{metodo_finanzas}" in hook: hook = hook.replace("{metodo_finanzas}", "un método probado")
                if "{porcentaje_finanzas}" in hook: hook = hook.replace("{porcentaje_finanzas}", "20")
                if "{nombre_metodo}" in hook: hook = hook.replace("{nombre_metodo}", "Warren Buffett")

                # General Technology
                if "{tecnologia_general}" in hook: hook = hook.replace("{tecnologia_general}", "la realidad virtual")
                if "{industria_general}" in hook: hook = hook.replace("{industria_general}", "la educación")
                if "{ProductoA_tech}" in hook: hook = hook.replace("{ProductoA_tech}", "iPhone 16")
                if "{ProductoB_tech}" in hook: hook = hook.replace("{ProductoB_tech}", "Galaxy S25")
                if "{producto_tech_review}" in hook: hook = hook.replace("{producto_tech_review}", random.choice(productos) if productos else "el nuevo smartphone")

                return hook
            
            # Fallback if theme is not defined
            return "Descubre cómo esto cambiará tu perspectiva para siempre."
        except Exception as e:
            st.error(f"Error en generar_hook_optimizado: {str(e)}")
            return "¡Esto es algo que no te puedes perder!"
            
# ======================
# AUXILIARY ADVANCED FUNCTIONS
# ======================

nlp = None 

@st.cache_resource
def get_spacy_model():
    """Loads the SpaCy model for Spanish."""
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        st.error("SpaCy model 'es_core_news_sm' not found. Attempting to download...")
        spacy.cli.download("es_core_news_sm")
        return spacy.load("es_core_news_sm")

def extraer_entidades(texto, tipo_entidad=None):
    """Extracts named entities (people, organizations, places, products) from a text using SpaCy."""
    if nlp is None: 
        st.error("Error: SpaCy model not loaded. Please contact support.")
        return []
    doc = nlp(texto)
    entidades = []
    for ent in doc.ents:
        if tipo_entidad is None or ent.label_ == tipo_entidad:
            entidades.append(ent.text)
    return list(set(entidades))

# ======================
# 3. UPDATED MAIN FUNCTIONS
# ======================
def analizar_tematica(texto):
    """Improved thematic detection with higher confidence per keyword."""
    scores = defaultdict(int)
    for tema, data in TEMATICAS.items():
        for palabra in data["palabras_clave"]:
            if re.search(rf"\b{palabra}\b", texto.lower()):
                scores[tema] += 1
    
    if not scores:
        return ("General", 0)
    
    mejor_tema, puntaje = max(scores.items(), key=lambda x: x[1])
    confianza = min(100, puntaje * 20) # Scale to percentage, 5 keywords = 100%
    return (mejor_tema, confianza)

def mejorar_script(script, tema, pre_generated_hook=None):
    """Enhances scripts for any theme with viral techniques."""
    # Detect if the script already has explicit timestamp markers
    segmentos_temporales = re.findall(r"(\(\d+-\d+\s*(?:segundos|s)\).*)", script, re.IGNORECASE)
    tiene_estructura = bool(segmentos_temporales)
    
    mejoras_por_tema = {
        "Robótica": {
            "transiciones": ["SFX: Sonido futurista activándose", "Corte rápido a detalle de mecanismo", "✨ MEJORA: Toma de Ameca expresando una emoción sutil"],
        },
        "Fútbol": {
            "transiciones": ["SFX: Hinchada rugiendo", "Slow motion de jugada clave", "✨ MEJORA: Gráfico animado de estadística de jugador"],
        },
        "Finanzas": {
            "transiciones": ["Gráfico animado de crecimiento/caída", "Zoom a cifras clave", "SFX: Sonido de calculadora o transacción"],
        },
        "Mascotas": {
            "transiciones": ["SFX: Sonido de risas o asombro", "Corte a cara de sorpresa del dueño", "Música divertida subiendo", "✨ MEJORA: Primer plano a la expresión traviesa de la mascota"],
        },
        "Fórmula 1": {
            "transiciones": ["SFX: Chirrido de neumáticos", "Cámara lenta del trompo", "✨ MEJORA: Toma en cabina del piloto reaccionando", "Corte rápido entre diferentes ángulos de la acción"],
        }
    }
    
    # Generic replacements for improvements and CTA
    reemplazos_genericos = {
        "{numero}": str(random.randint(10, 60)),
        "{cantidad}": str(random.randint(1, 10)),
        "{pregunta}": "¿Qué te pareció?", # for generic CTA
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
            script_final_mejorado.append(linea) # Always include the original line
            
            # If the line contains a temporal marker, add a visual improvement after it
            if re.search(r"^\(\d+-\d+\s*(?:segundos|s)\)", linea, re.IGNORECASE):
                mejora_opciones = mejoras_por_tema.get(tema, {}).get("transiciones")
                if mejora_opciones:
                    mejora = random.choice(mejora_opciones)
                else: # Fallback to generic visual improvements
                    mejora = random.choice(plantillas_genericas["mejora_visual"])
                
                # Apply generic replacements to the improvement
                for k, v in reemplazos_genericos.items():
                    mejora = mejora.replace(k, v)
                
                # Add the improvement
                script_final_mejorado.append(f"✨ MEJORA: {mejora}")
                
        # At the end of the structured script, add a CTA if not already included
        # We check if an obvious CTA is already present in the last few lines of the *original* script
        cta_already_present_in_original = any(re.search(r"(comenta|suscribe|siguenos|cta|subscribe)", l.lower()) for l in script.split('\n')[-7:])

        if not cta_already_present_in_original:
            llamado_accion_gen = random.choice(plantillas_genericas["llamado_accion"])
            for k, v in reemplazos_genericos.items():
                llamado_accion_gen = llamado_accion_gen.replace(k, v)
            script_final_mejorado.append(f"\n(FINAL) 📲 LLAMADA A LA ACCIÓN: {llamado_accion_gen}")
            script_final_mejorado.append(f"✨ SUGERENCIA VISUAL: Considera añadir cortes rápidos y música dinámica.")
            
    else: # If the script has NO explicit temporal structure
        # Use the pre-generated hook if it exists, otherwise generate a generic one
        gancho_a_usar = pre_generated_hook if pre_generated_hook else generar_hook(tema, reemplazos_genericos)
        llamado_accion_gen = random.choice(plantillas_genericas["llamado_accion"])
        for k, v in reemplazos_genericos.items():
            llamado_accion_gen = llamado_accion_gen.replace(k, v)
        
        script_final_mejorado.append(f"(0-5 segundos) 🎯 GANCHO INICIAL: {gancho_a_usar}")
        script_final_mejorado.append("\n" + script.strip() + "\n") # Insert the entire original script
        script_final_mejorado.append(f"(FINAL) 📲 LLAMADA A LA ACCIÓN: {llamado_accion_gen}")
        script_final_mejorado.append(f"✨ SUGERENCIA VISUAL: Considera añadir cortes rápidos y música dinámica.")

    script_final = '\n'.join(script_final_mejorado)
    
    # Add hashtags at the end
    hashtags = TEMATICAS.get(tema, {}).get("hashtags", ["#Viral", "#Trending"])
    script_final += f"\n\n🔖 HASHTAGS: {' '.join(hashtags[:4])}" # Limit to 4 hashtags for greater impact
    
    return script_final

def generar_hook(tema, reemplazos):
    """Generates dynamic thematic hooks (used as fallback or for generic hooks)."""
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
    
    # Apply general replacements to the hooks generated here
    for k, v in reemplazos.items():
        hook = hook.replace(k, v)
    
    return hook

# ======================
# 4. OPTIMIZED STREAMLIT INTERFACE
# ======================

@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK resources."""
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
    nlp = get_spacy_model() # Load the SpaCy model at startup

    # Train the hook optimizer with relevant examples
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
        "El error de Hamilton que le costó la carrera"
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
                    
                    # Generate the main hook here
                    generated_hook = hook_ai.generar_hook_optimizado(texto, tema) 
                    
                    # Pass the main hook to mejorar_script
                    script_mejorado = mejorar_script(texto, tema, pre_generated_hook=generated_hook)
                    
                    # Hashtags are now generated within mejorar_script, only displayed here
                    hashtags_display = ' '.join(TEMATICAS.get(tema, {}).get("hashtags", ["#Viral"]))

                    st.subheader(f"🎯 Temática: **{tema}** (Confianza: **{confianza}%**)")
                    
                    st.text_area("Hook Viral Recomendado:", value=generated_hook, height=100) 
                    
                    st.text_area("Script Optimizado:", value=script_mejorado, height=450) # Increased size
                    
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
