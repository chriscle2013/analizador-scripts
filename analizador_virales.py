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
        "palabras_clave": ["humanoide", "bípedo", "androide", "atlas", "asimo"],
        "hooks": {
            "técnica": ["Los desafíos de la **locomoción bípeda** en {nombre_robot}"],
            "aplicación": ["Cómo los humanoides están revolucionando la {industria}"],
            "avance": ["El nuevo sensor de {compañía} que permite a los humanoides {acción_mejorada}"]
        },
        "hashtags": ["#Humanoides", "#RobotsHumanoides", "#Bípedos"]
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
# 2. SISTEMA DE APRENDIZAJE AUTOMÁTICO ROBUSTO
# ======================
class HookOptimizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = None
        self.hooks_db = []
        
    def entrenar(self, hooks_virales):
        """Versión mejorada con manejo de errores"""
        if not hooks_virales or len(hooks_virales) < 2:
            return False
            
        try:
            X = self.vectorizer.fit_transform(hooks_virales)
            self.model = KMeans(n_clusters=min(3, len(hooks_virales)-1))
            self.model.fit(X)
            self.hooks_db = hooks_virales
            return True
        except Exception as e:
            st.error(f"Error en entrenamiento: {str(e)}")
            return False
        
    def generar_hook_optimizado(self, texto, tema):
        """Genera hooks contextuales"""
        try:
            # Priorizar hooks de la temática detectada
            if tema in TEMATICAS:
                hooks_tema = TEMATICAS[tema]["hooks"]
                estrategia = random.choice(list(hooks_tema.keys()))
                plantilla = random.choice(hooks_tema[estrategia])
                hook = plantilla.replace("{robot}", "Ameca") \
                               .replace("{tecnología}", "robótica") \
                               .replace("{industria}", "la interacción humano-máquina")
                return hook
            
            # Fallback para temas no definidos
            return "Descubre cómo esta tecnología está cambiando el mundo"
        except:
            return "Este contenido te sorprenderá"

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
    confianza = min(100, puntaje * 20)  # Escala a porcentaje
    return (mejor_tema, confianza)

def mejorar_script(script, tema):
    """Mejora scripts para cualquier temática con técnicas virales"""
    # 1. Detección de estructura existente
    segmentos_temporales = re.findall(r"(\(\d+-\d+\ssegundos\).*)", script)
    tiene_estructura = bool(segmentos_temporales)
    
    # 2. Diccionario de mejoras por temática (usando get() para default)
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
        # ... otros temas ...
    }
    
    # 3. Sistema de reemplazos dinámicos
    reemplazos = {
        "{año}": str(datetime.now().year),
        "{robot}": "Ameca" if tema == "Robótica" else "este dispositivo",
        "{jugador}": random.choice(["Messi", "Cristiano", "Haaland"]) if tema == "Fútbol" else "el protagonista",
        # ... otros reemplazos ...
    }
    
    # 4. Plantillas multiuso (cuando no hay datos específicos)
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

    # 5. Procesamiento del script
    if tiene_estructura:
        # Mejorar script estructurado
        lineas = script.split('\n')
        script_mejorado = []
        
        for linea in lineas:
            script_mejorado.append(linea)
            
            # Añadir mejoras después de cada segmento temporal
            if any(sec in linea for sec in ["(0-3 segundos)", "(3-10 segundos)", "(10-30 segundos)"]):
                # Seleccionar mejora adecuada al tema
                mejora = (
                    random.choice(mejoras_por_tema.get(tema, {}).get("transiciones", [])) or
                    random.choice(plantillas_genericas["mejora_visual"])
                
                # Aplicar reemplazos
                for k, v in reemplazos.items():
                    mejora = mejora.replace(k, v)
                
                script_mejorado.append(f"✨ MEJORA: {mejora}")
                
    else:
        # Reestructurar script no organizado
        frases = [f.strip() for f in re.split(r'[.!?]', script) if f.strip()]
        
        estructura_base = [
            "(0-3 segundos) 🎯 GANCHO INICIAL",
            frases[0] if frases else generar_hook(tema, reemplazos),
            "(3-10 segundos) 💡 BENEFICIO CLAVE",
            ' '.join(frases[1:3]) if len(frases) > 2 else "Descubre cómo...",
            "(10-30 segundos) 🚀 DESARROLLO",
            ' '.join(frases[3:5]) if len(frases) > 4 else "La innovación continúa...",
            "(30-35 segundos) 📲 INTERACCIÓN",
            random.choice(plantillas_genericas["llamado_accion"])
        ]
        
        script_mejorado = estructura_base

    # 6. Post-procesamiento (aplicar a todo el script)
    script_final = '\n'.join(script_mejorado) if isinstance(script_mejorado, list) else script_mejorado
    
    # Añadir hashtags al final
    hashtags = TEMATICAS.get(tema, {}).get("hashtags", ["#Viral", "#Trending"])
    script_final += f"\n\n🔖 HASHTAGS: {' '.join(hashtags[:3])}"
    
    return script_final

def generar_hook(tema, reemplazos):
    """Genera hooks temáticos dinámicos"""
    hooks_disponibles = [
        TEMATICAS.get(tema, {}).get("hooks", {}),
        {
            "impacto": ["Lo que nadie te dijo sobre {tema}"],
            "curiosidad": ["¿Por qué {tema} está revolucionando todo?"]
        }
    ]
    
    # Seleccionar hook aleatorio aplicando reemplazos
    hook = random.choice(
        hooks_disponibles[0].get(random.choice(list(hooks_disponibles[0].keys())), []) +
        hooks_disponibles[1].get(random.choice(list(hooks_disponibles[1].keys())), [])
    )
    
    for k, v in reemplazos.items():
        hook = hook.replace(k, v)
    
    return hook if hook else "Descubre esto que cambiará tu perspectiva"

# ======================
# 4. INTERFAZ STREAMLIT OPTIMIZADA
# ======================
def main():
    st.set_page_config(layout="wide", page_title="🔥 ViralHook Generator PRO")
    
    # Inicializar sistemas
    hook_ai = HookOptimizer()
    hook_ai.entrenar([
        "Cómo los robots como Ameca están cambiando la industria",
        "La evolución de los humanoides en 2024",
        "Ameca vs humanos: ¿Quién es más expresivo?",
        "Esta tecnología robótica te sorprenderá"
    ])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎬 Script para Análizar")
        texto = st.text_area("Pega tu script completo:", height=300,
                           placeholder="Ej: (0-3 segundos) Video impactante...")
        
    with col2:
        if st.button("🚀 Optimizar Contenido"):
            if texto:
                with st.spinner("Analizando y mejorando..."):
                    # Análisis avanzado
                    tema, confianza = analizar_tematica(texto)
                    blob = TextBlob(texto)
                    polaridad = blob.sentiment.polarity
                    
                    # Generación de contenido
                    hook = hook_ai.generar_hook_optimizado(texto, tema)
                    script_mejorado = mejorar_script(texto, tema)
                    hashtags = ' '.join(TEMATICAS.get(tema, {}).get("hashtags", ["#Viral"]))
                    
                    # Mostrar resultados
                    st.subheader(f"🎯 Temática: {tema} (Confianza: {confianza}%)")
                    st.text_area("Hook Viral Recomendado:", value=hook, height=100)
                    st.text_area("Script Optimizado:", value=script_mejorado, height=300)
                    
                    # Métricas
                    with st.expander("📊 Análisis Avanzado"):
                        st.metric("Sentimiento", 
                                "🔥 Positivo" if polaridad > 0.1 else "😐 Neutral" if polaridad > -0.1 else "⚠️ Negativo",
                                delta=f"{polaridad:.2f}")
                        st.write(f"🔍 Hashtags recomendados: {hashtags}")
            else:
                st.warning("Por favor ingresa un script para analizar")

if __name__ == "__main__":
    main()
