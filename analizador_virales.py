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

# --- ¡CRUCIAL! st.set_page_config DEBE ser la primera llamada a un comando de Streamlit. ---
st.set_page_config(layout="wide", page_title="🔥 ViralHook Generator PRO")

# ======================
# 1. BASE DE DATOS DE TEMÁTICAS
# ======================
TEMATICAS = {
    # Deportes
    "Fórmula 1": {
        "palabras_clave": ["f1", "gran premio", "piloto", "carrera", "escudería", "circuito", "clasificación", "trompos", "silverstone", "ferrari", "mercedes", "red bull", "aston martin", "alpine", "alonso", "hamilton", "verstappen", "sainz", "leclerc", "pole position", "pole", "vuelta rápida", "victoria", "podio", "adelantamiento", "campeón"], 
        "hooks": {
            "técnica": ["El {sistema} que hizo a {equipo_f1} ganar en {circuito_f1}"],
            "polémica": ["La decisión de la FIA que cambió el {evento_f1_generico}"],
            "récord": ["{piloto_f1} rompió el récord de {marca} en {año}", "La {marca} que le dio la {logro_f1} a {piloto_f1}"], # Añadido
            "inesperado": ["¡El {evento_f1_inesperado} más caótico en la historia de {circuito_f1}!", "Los {numero} trompos más salvajes de {evento_f1_generico}"],
            "logro": ["La {logro_f1} de {piloto_f1} que te dejará sin aliento en {circuito_f1}", "Así fue la {logro_f1} de {piloto_f1} en la última vuelta", "La {marca} que le dio la {logro_f1} a {piloto_f1}"] # Añadido
        },
        "hashtags": ["#F1", "#Formula1", "#F1News", "#SilverstoneF1", "#MotorSport", "#F1Pole"] 
    },
    "Fútbol": {
        "palabras_clave": ["gol", "partido", "jugador", "liga", "champions", "equipo", "copa", "mundial", "messi", "ronaldo", "mbappe", "club", "delantero", "defensa", "portero", "entrenador", "penalti"],
        "hooks": {
            "táctica": ["El {sistema_juego} que hizo campeón a {equipo_futbol}"],
            "polémica": ["El {incidente} más injusto de la historia del fútbol"],
            "dato": ["{jugador_futbol} tiene este récord de {estadística_futbol}"]
        },
        "hashtags": ["#Fútbol", "#Champions", "#LaLiga", "#FPC", "#Futebol"]
    },

    # Robótica
    "Robots Humanoides": {
        "palabras_clave": ["humanoide", "bípedo", "androide", "atlas", "asimo", "ameca", "engineered arts", "robot", "optimus", "tesla bot", "autónomo", "ia", "inteligencia artificial", "movimiento", "futuro"],
        "hooks": {
            "técnica": ["Los desafíos de la **locomoción bípeda** en {nombre_robot}"],
            "aplicación": ["Cómo los humanoides están revolucionando la {industria_robotica}"],
            "avance": ["El nuevo sensor de {compañia_robotica} que permite a los humanoides {accion_mejorada_robotica}"],
            "impacto": ["Así es {nombre_robot}, el robot humanoide que cambiará el mundo"], 
            "demo": ["Mira a {nombre_robot} haciendo ESTO en el laboratorio de {compañia_robotica}", "Las increíbles capacidades de {nombre_robot} que te dejarán sin palabras"] 
        },
        "hashtags": ["#Humanoides", "#RobotsHumanoides", "#Bípedos", "#Ameca", "#Optimus", "#TeslaBot", "#FuturoAI", "#Robótica"]
    },
    "Inteligencia Artificial en Robótica": {
        "palabras_clave": ["ia", "aprendizaje automático", "machine learning", "visión artificial", "deep learning", "algoritmos", "inteligencia artificial", "red neuronal", "datos", "autonomía", "percepción"],
        "hooks": {
            "técnica": ["La **red neuronal** que permite a {robot_ia} reconocer {objeto_ia}"],
            "aplicación": ["IA para la **navegación autónoma** en {entorno_complejo}"],
            "impacto": ["Cómo el {algoritmo_ia} está optimizando el {proceso_robotico}"]
        },
        "hashtags": ["#AIRobótica", "#IA", "#MachineLearningRobots", "#VisiónArtificial", "#DeepLearning"]
    },
    "Robots Colaborativos (Cobots)": {
        "palabras_clave": ["cobots", "colaborativos", "seguridad", "interacción h-r", "industria 4.0", "manufactura", "fábrica", "producción", "brazo robótico"],
        "hooks": {
            "beneficio": ["**Cobots**: Mejorando la {productividad} y la {seguridad} en {sector_industrial}"],
            "implementación": ["Desafíos y soluciones al integrar **cobots** en {tipo_empresa}"],
            "futuro": ["El rol de los **robots colaborativos** en la {proxima_decada}"]
        },
        "hashtags": ["#Cobots", "#RobotsColaborativos", "#Industria40", "#Automatización"]
    },
    "Robótica Médica": {
        "palabras_clave": ["cirugía robótica", "quirúrgico", "da vinci", "rehabilitación", "exosqueletos", "telemedicina", "salud", "hospital", "paciente", "diagnóstico", "asistencia"],
        "hooks": {
            "innovación": ["**Robótica médica**: La precisión de {sistema_robotico_medico} en {procedimiento_medico}"],
            "impacto_paciente": ["Cómo los **exosqueletos** están transformando la {condicion_paciente}"],
            "futuro": ["La próxima generación de **robots asistenciales** en {ambito_salud}"]
        },
        "hashtags": ["#RobóticaMédica", "#CirugíaRobótica", "#Exoesqueletos", "#SaludDigital", "#Medicina"]
    },

    # Mascotas
    "Mascotas": {
        "palabras_clave": ["perro", "gato", "hámster", "pájaro", "mascota", "animales", "cachorros", "golden retriever", "labrador", "loro", "periquito", "conejo", "hurón", "cola", "patas", "dueño", "veterinario", "juguetes", "cajas", "morder", "arañar", "pelaje", "ronroneo"],
        "hooks": {
            "humor": ["Tu mascota también hace ESTO para volverte loco", "¿Listo para reírte? Las travesuras más épicas de {animal_mascota}"],
            "consejo": ["El secreto para que tu {tipo_mascota} deje de {mal_habito_mascota}"],
            "emocional": ["La historia de {animal_mascota} que te derretirá el corazón"]
        },
        "hashtags": ["#Mascotas", "#AnimalesGraciosos", "#MascotasVirales", "#PetsOfTikTok", "#AmorAnimal", "#Gatos", "#Perros"]
    },

    # Mindset
    "Mindset": {
        "palabras_clave": ["éxito", "hábitos", "mentalidad", "crecimiento", "productividad", "motivación", "superación", "bienestar", "felicidad", "procrastinación", "metas", "disciplina"],
        "hooks": {
            "científico": ["Estudio de Harvard prueba que {hábito_mindset} aumenta {metrica_mindset}"],
            "inspiración": ["Cómo {persona_mindset} pasó de {situacion_mindset} a {logro_mindset}"],
            "acción": ["Si haces esto cada mañana, tu vida cambiará en {tiempo_mindset}"]
        },
        "hashtags": ["#Mindset", "#CrecimientoPersonal", "#Motivacion", "#Productividad", "#DesarrolloPersonal"]
    },

    # Finanzas
    "Finanzas": {
        "palabras_clave": ["dinero", "inversión", "ahorro", "finanzas", "criptomonedas", "bolsa", "negocios", "emprendimiento", "bitcoin", "acciones", "mercado", "dólar", "euros", "impuestos", "ingresos", "gastos"],
        "hooks": {
            "impacto": ["Cómo ahorré {cantidad_dinero} en {tiempo_finanzas} con {metodo_finanzas}"],
            "error": ["El error que te hace perder {porcentaje_finanzas}% de tus ingresos"],
            "sistema": ["El método {nombre_metodo} para multiplicar tu dinero"]
        },
        "hashtags": ["#Finanzas", "#Ahorro", "#Inversión", "#Dinero", "#Emprendimiento", "#Cripto"]
    },

    # Tecnología (General)
    "Tecnología": {
        "palabras_clave": ["robot", "ia", "tecnología", "automatización", "innovación", "gadget", "futuro", "ciencia", "dispositivo", "software", "hardware", "app", "metaverso", "realidad virtual", "ciberseguridad"],
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
        """Genera hooks contextuales usando detección de entidades del script y la temática."""
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
                
                # Lógica mejorada para seleccionar la estrategia del hook
                estrategia = random.choice(list(hooks_tema.keys())) # Opción por defecto
                
                if tema == "Fórmula 1":
                    if re.search(r'\b(trompos|spin|accidente|caos|inesperado)\b', texto.lower()) and "inesperado" in hooks_tema:
                        estrategia = "inesperado"
                    elif re.search(r'\b(pole|victoria|ganar|récord|campeón|última vuelta)\b', texto.lower()) and ("logro" in hooks_tema or "récord" in hooks_tema):
                        estrategia = random.choice(["logro", "récord"]) 
                    elif re.search(r'\b(polémica|fia|sanción|protesta)\b', texto.lower()) and "polémica" in hooks_tema:
                        estrategia = "polémica"
                elif tema == "Mascotas":
                    if re.search(r'\b(chistoso|gracioso|divertido|travesuras|humor)\b', texto.lower()) and "humor" in hooks_tema:
                        estrategia = "humor"
                elif tema == "Robots Humanoides": # Para el caso de Optimus
                    if re.search(r'\b(optimus|tesla bot|movimiento|precisión|eficiencia|futuro)\b', texto.lower()) and ("impacto" in hooks_tema or "demo" in hooks_tema):
                        estrategia = random.choice(["impacto", "demo"])
                
                plantilla = random.choice(hooks_tema.get(estrategia, list(hooks_tema.values())[0])) # Fallback si la estrategia no existe

                hook = plantilla
                
                # REEMPLAZOS ESPECÍFICOS POR TEMÁTICA
                if tema == "Fórmula 1":
                    if "{piloto_f1}" in hook: hook = hook.replace("{piloto_f1}", random.choice(f1_pilotos) if f1_pilotos else random.choice(["Verstappen", "Hamilton", "Leclerc"]))
                    if "{equipo_f1}" in hook: hook = hook.replace("{equipo_f1}", random.choice(f1_equipos) if f1_equipos else random.choice(["Red Bull", "Ferrari", "Mercedes"]))
                    if "{circuito_f1}" in hook: hook = hook.replace("{circuito_f1}", random.choice(f1_circuitos) if f1_circuitos else random.choice(["Silverstone", "Mónaco", "Spa"]))
                    if "{evento_f1_generico}" in hook: hook = hook.replace("{evento_f1_generico}", random.choice(["Gran Premio", "clasificación", "carrera", "Q3"]))
                    if "{evento_f1_inesperado}" in hook: hook = hook.replace("{evento_f1_inesperado}", random.choice(["Gran Premio", "sesión de clasificación"]))
                    if "{numero}" in hook: hook = hook.replace("{numero}", str(random.randint(3, 10))) 
                    if "{marca}" in hook: hook = hook.replace("{marca}", random.choice(["velocidad récord", "tiempo más rápido", "vuelta imbatible"]))
                    if "{logro_f1}" in hook: hook = hook.replace("{logro_f1}", random.choice(["Pole Position", "victoria épica", "vuelta de la vida"]))

                elif tema == "Mascotas":
                    if "{animal_mascota}" in hook:
                        if nombres_animales_en_script:
                            hook = hook.replace("{animal_mascota}", random.choice(nombres_animales_en_script))
                        elif personas: 
                            hook = hook.replace("{animal_mascota}", random.choice(personas))
                        else: 
                            hook = hook.replace("{animal_mascota}", random.choice(["tu adorable mascota", "este peludo amigo", "este travieso animal"]))

                    if "{tipo_mascota}" in hook:
                        if nombres_animales_en_script:
                            hook = hook.replace("{tipo_mascota}", random.choice(nombres_animales_en_script))
                        else: 
                            hook = hook.replace("{tipo_mascota}", random.choice(["perro", "gato", "loro", "hámster"]))

                    if "{mal_habito_mascota}" in hook:
                        hook = hook.replace("{mal_habito_mascota}", random.choice(["ladrar mucho", "arañar muebles", "morder cables", "comerse los zapatos"]))

                # Robotics (Optimus)
                if tema == "Robots Humanoides":
                    robot_name = random.choice(productos) if productos else "Optimus"
                    company_name = random.choice(organizaciones) if organizaciones else "Tesla"
                    
                    # CORRECCIÓN: Evitar "en el laboratorio de Optimus" si Optimus es el robot
                    if "{compañia_robotica}" in hook:
                        if company_name.lower() == robot_name.lower(): # Si el nombre del robot y la compañía son iguales o muy similares
                            hook = hook.replace("{compañia_robotica}", "su creador" if "tesla" not in texto.lower() else "Tesla") # Usar "su creador" o "Tesla"
                        else:
                            hook = hook.replace("{compañia_robotica}", company_name)
                    if "{nombre_robot}" in hook: hook = hook.replace("{nombre_robot}", robot_name)
                    
                    if "{industria_robotica}" in hook: hook = hook.replace("{industria_robotica}", "la manufactura")
                    if "{accion_mejorada_robotica}" in hook: hook = hook.replace("{accion_mejorada_robotica}", "navegar con destreza")


                # REEMPLAZOS GENÉRICOS (for placeholders that can appear in multiple themes)
                if "{año}" in hook: hook = hook.replace("{año}", str(datetime.now().year + 1)) 
                if "{marca}" in hook: hook = hook.replace("{marca}", "velocidad récord")
                if "{incidente}" in hook: hook = hook.replace("{incidente}", "incidente polémico")
                if "{dato_impactante}" in hook: hook = hook.replace("{dato_impactante}", "un dato sorprendente")
                if "{novedad}" in hook: hook = hook.replace("{novedad}", "la última novedad")
                if "{pregunta}" in hook: hook = hook.replace("{pregunta}", "¿Estás de acuerdo?")
                if "{tema}" in hook: hook = hook.replace("{tema}", tema)
                if "{sistema}" in hook: hook = hook.replace("{sistema}", "sistema secreto") 

                # General Robotics / AI
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
# FUNCIONES AUXILIARES AVANZADAS
# ======================

@st.cache_resource
def get_spacy_model():
    """Carga el modelo de SpaCy para español."""
    try:
        # Aseguramos que solo se intente cargar, no descargar aquí.
        # Las descargas se manejan en download_nltk_data o como parte de la configuración del entorno.
        return spacy.load("es_core_news_sm")
    except OSError:
        st.error("Modelo 'es_core_news_sm' de SpaCy no encontrado. Asegúrate de que esté instalado en tu entorno.")
        # No intentamos descargar aquí para evitar problemas de permisos o bloqueos en Streamlit Cloud.
        # La descarga debería hacerse al construir la imagen de Docker o en un script de pre-ejecución.
        return None # Devuelve None si no se puede cargar el modelo

@st.cache_resource
def download_nltk_data():
    """Descarga los recursos de NLTK necesarios."""
    nltk_data_path = nltk.data.find('corpora') # Directorio por defecto para descargas de NLTK
    
    # Lista de paquetes NLTK requeridos
    required_nltk_packages = ['punkt', 'averaged_perceptron_tagger']

    for package in required_nltk_packages:
        try:
            # Intenta encontrar el paquete. Si no lo encuentra, lanza una excepción.
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'taggers/{package}')
        except nltk.downloader.DownloadError:
            st.warning(f"Descargando paquete NLTK: {package}...")
            try:
                nltk.download(package, quiet=True) # quiet=True para no mostrar barra de progreso
                st.success(f"Paquete NLTK '{package}' descargado con éxito.")
            except Exception as e:
                st.error(f"Error al descargar NLTK '{package}': {str(e)}. Por favor, verifica tu conexión o permisos.")
        except Exception as e:
            st.error(f"Error inesperado al verificar NLTK '{package}': {str(e)}")
            
# Inicializa 'nlp' una única vez al cargar el script.
# La llamada a st.set_page_config ya ocurrió.
nlp = get_spacy_model()

def extraer_entidades(texto, tipo_entidad=None):
    """Extrae entidades nombradas (personas, organizaciones, lugares, productos) de un texto usando SpaCy."""
    if nlp is None: 
        st.error("Error: Modelo de SpaCy no cargado. Por favor, asegúrate de que 'es_core_news_sm' esté instalado y contacta al soporte si el problema persiste.")
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
    """Mejora scripts para cualquier temática con técnicas virales."""
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
    """Genera hooks para una temática dada."""
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

# ======================
# 4. INTERFAZ STREAMLIT OPTIMIZADA
# ======================
    
def main():
    # Asegúrate de que las descargas de NLTK y la carga del modelo SpaCy 
    # se hagan *después* de st.set_page_config() y solo una vez.
    download_nltk_data() # Ahora maneja mejor los errores y descargas

    # nlp se inicializa globalmente después de set_page_config
    if nlp is None:
        st.error("El modelo de SpaCy no pudo ser cargado. La funcionalidad de análisis de entidades será limitada.")

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
        
        # El estado de sesión para script_content debe inicializarse *antes* de usarse
        if 'script_content' not in st.session_state:
            st.session_state.script_content = ""

        texto = st.text_area("Pega tu script completo:", 
                             height=300,
                             placeholder="Ej: (0-3 segundos) Video impactante...",
                             key="script_input_area", 
                             value=st.session_state.script_content) 
        
        # Botón de borrar (lo incluí de nuevo porque dices que antes funcionaba con él y puede ser útil)
        if st.button("🗑️ Borrar Script", key="clear_script_button"):
            st.session_state.script_content = "" 
            st.experimental_rerun() 

    with col2:
        if st.button("🚀 Optimizar Contenido"):
            if texto: 
                with st.spinner("Analizando y mejorando..."):
                    tema, confianza = analizar_tematica(texto)
                    
                    # Asegúrate de que TextBlob y NRCLex también estén bien manejados
                    try:
                        blob = TextBlob(texto) 
                        polaridad = blob.sentiment.polarity
                    except Exception as e:
                        st.warning(f"No se pudo realizar el análisis de sentimiento: {e}. Continuado sin este análisis.")
                        polaridad = 0.0 # Default a neutral
                        
                    try:
                        emotions = NRCLex(texto).affect_frequencies
                    except Exception as e:
                        st.warning(f"No se pudo realizar el análisis de emociones: {e}. Continuado sin este análisis.")
                        emotions = {}

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

                        st.subheader("Emociones Detectadas:")
                        if emotions:
                            emociones_relevantes = {k: v for k, v in emotions.items() if v > 0.05} 
                            if emociones_relevantes:
                                for emotion, freq in sorted(emociones_relevantes.items(), key=lambda item: item[1], reverse=True):
                                    st.write(f"- **{emotion.capitalize()}**: {freq:.2%}")
                            else:
                                st.write("No se detectaron emociones fuertes en el script.")
                        else:
                            st.write("Análisis de emociones no disponible debido a un error previo.")


                        st.write(f"🔍 Hashtags recomendados: {hashtags_display}")
            else:
                st.warning("Por favor ingresa un script para analizar")

if __name__ == "__main__":
    main()
