
import streamlit as st
from textblob import TextBlob
import re
import nltk
import random

# Asegurar recursos para TextBlob
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Función para analizar sentimiento
def detectar_sentimiento(texto):
    blob = TextBlob(texto)
    sentimiento = blob.sentiment.polarity
    if sentimiento > 0.1:
        return "positivo"
    elif sentimiento < -0.1:
        return "negativo"
    else:
        return "neutro"

# Evaluación de viralidad
def evaluar_viralidad(script):
    puntaje = 0
    if any(p in script.lower() for p in ["sabías que", "no vas a creer", "te cuento algo", "impactante", "emocionante"]):
        puntaje += 2
    if any(p in script.lower() for p in ["guárdalo", "comparte", "dale like", "comenta"]):
        puntaje += 1
    if len(script.split()) < 80:
        puntaje += 1
    if puntaje >= 3:
        return "Alto"
    elif puntaje == 2:
        return "Medio"
    else:
        return "Bajo"

# Recomendaciones
def generar_recomendaciones(script):
    recomendaciones = []
    if not any(p in script.lower() for p in ["sabías que", "no vas a creer", "te cuento algo"]):
        recomendaciones.append("Agrega un hook llamativo al inicio.")
    if any(len(frase.split()) > 20 for frase in script.split(".")):
        recomendaciones.append("Reduce la longitud de las frases para mantener la atención.")
    if not any(p in script.lower() for p in ["guárdalo", "comparte", "comenta", "sígueme"]):
        recomendaciones.append("Incluye una llamada a la acción (CTA).")
    return recomendaciones

# Detección de temática
def detectar_tematica(script):
    script = script.lower()
    temas = {
        "motivación": ["lograr", "superar", "esfuerzo", "disciplina", "motivación", "reto", "crecimiento", "cambiar", "vida", "metas"],
        "autoestima": ["valor", "mereces", "amor propio", "autoestima", "confianza", "creer en ti", "aceptación"],
        "dinero": ["dinero", "finanzas", "abundancia", "deuda", "rico", "pobre", "ahorro", "invertir", "negocio"],
        "salud": ["salud", "bienestar", "ejercicio", "dormir", "hábitos", "comida", "cuerpo", "energía", "alimentación"],
        "relaciones": ["pareja", "amor", "ruptura", "tóxica", "relaciones", "familia", "soledad", "amistad"],
        "éxito": ["éxito", "logro", "trabajo", "profesional", "empresa", "emprender", "meta", "sueño"],
        "espiritualidad": ["espíritu", "alma", "universo", "vibración", "Dios", "meditación", "energía", "consciencia"]
    }
    for tema, palabras in temas.items():
        if any(p in script for p in palabras):
            return tema
    return "general"

# Generar hook
def generar_hook(script):
    sentimiento = detectar_sentimiento(script)
    tematica = detectar_tematica(script)
    hooks = {
        "motivación": {
            "positivo": ["Lo que estás por leer puede encender tu fuego interior.", "Estás más cerca de lo que imaginas, y este mensaje te lo recordará."],
            "negativo": ["Cuando todo parece difícil, este mensaje puede darte fuerza.", "A veces perderse es parte de encontrarse."],
            "neutro": ["Un cambio pequeño puede iniciar una gran transformación.", "Este mensaje es un punto de partida."]
        },
        "autoestima": {
            "positivo": ["Esto te recordará lo mucho que vales.", "Solo necesitas creer en ti. Este mensaje te lo muestra."],
            "negativo": ["¿Te cuesta quererte? Esto puede ayudarte.", "Hay una parte de ti que olvidaste... este mensaje la despierta."],
            "neutro": ["Una pausa para reconectar contigo mismo.", "Este texto es un espejo. Mírate con amor."]
        },
        "dinero": {
            "positivo": ["Este consejo puede ayudarte a atraer más abundancia.", "Lo que estás por leer puede transformar tu relación con el dinero."],
            "negativo": ["¿Te cuesta manejar el dinero? Esto puede ayudarte.", "Este mensaje puede darte claridad financiera cuando más lo necesitas."],
            "neutro": ["Una reflexión sobre el dinero que podría cambiar tu mentalidad.", "Esto puede ser clave para tu estabilidad financiera."]
        },
        "salud": {
            "positivo": ["Tu bienestar empieza con pequeños pasos. Aquí va uno.", "Esto puede ser el inicio de un cambio saludable."],
            "negativo": ["Tal vez estás descuidando lo más valioso: tu salud.", "A veces ignoramos las señales... este mensaje es una de ellas."],
            "neutro": ["Una dosis de conciencia sobre tu salud.", "Esto que leerás puede ayudarte a mejorar tu energía."]
        },
        "relaciones": {
            "positivo": ["Esto puede mejorar la forma en que te conectas con los demás.", "Amar también es aprender. Este mensaje lo resume."],
            "negativo": ["Si estás pasando por un mal momento emocional, esto es para ti.", "A veces lo más duro que escuchamos es lo que más necesitamos."],
            "neutro": ["Una mirada clara sobre el amor y las relaciones.", "Este mensaje puede ayudarte a entender tus vínculos."]
        },
        "éxito": {
            "positivo": ["Esto puede impulsarte hacia tus metas.", "El éxito empieza por lo que estás a punto de leer."],
            "negativo": ["Fracasar también es parte del camino. Este mensaje te lo explica.", "Si te sientes estancado, esto puede ayudarte."],
            "neutro": ["Una reflexión que puede cambiar tu forma de alcanzar el éxito.", "Este mensaje puede redirigir tu ambición."]
        },
        "espiritualidad": {
            "positivo": ["Estás donde necesitas estar. Este mensaje te lo confirmará.", "Lo que leerás ahora puede elevar tu energía."],
            "negativo": ["A veces el alma necesita palabras más que el cuerpo.", "Este mensaje puede guiarte cuando sientas que te perdiste."],
            "neutro": ["Un llamado sutil a tu interior.", "Esto puede ayudarte a reconectar con tu esencia."]
        },
        "general": {
            "positivo": ["Este mensaje puede inspirarte más de lo que imaginas.", "Lo que estás por leer puede motivarte a actuar."],
            "negativo": ["Esto puede doler, pero lo necesitas.", "Hay verdad en estas palabras. Escúchalas con el corazón."],
            "neutro": ["Una idea que puede cambiar tu día.", "Esto podría abrirte una nueva perspectiva."]
        }
    }
    return random.choice(hooks[tematica][sentimiento])

# Mejorar el script
def mejorar_script(script):
    primeras_lineas = script.strip().split('
')[0][:120].lower()
    if not any(p in primeras_lineas for p in ["sabías que", "no vas a creer", "te cuento algo", "esto te va a sorprender", "te ha pasado que", "lo que estás por leer", "esto puede", "este mensaje"]):
        hook = generar_hook(script) + "
"
    else:
        hook = ""

    frases = script.split('.')
    frases_mejoradas = []
    for frase in frases:
        frase = frase.strip()
        if frase:
            palabras = frase.split()
            if len(palabras) > 20:
                mitad = len(palabras) // 2
                primera = " ".join(palabras[:mitad])
                segunda = " ".join(palabras[mitad:])
                frases_mejoradas.append(primera + '.')
                frases_mejoradas.append(segunda + '.')
            else:
                frases_mejoradas.append(frase + '.')

    palabras_emocionales = ['increíble', 'impactante', 'motivador', 'emocionante']
    if not any(p in script.lower() for p in palabras_emocionales):
        frases_mejoradas.append("Este mensaje es tan poderoso que puede inspirarte de verdad.")

    if not re.search(r"(sígueme|dale like|guárdalo|comenta|etiqueta)", script.lower()):
        frases_mejoradas.append("💬 Comenta si te hizo sentido y guárdalo para recordarlo.")

    nuevo_script = hook + "
".join(frases_mejoradas)
    return nuevo_script

# Interfaz Streamlit
st.title("🎬 Analizador de Reels Virales")
st.markdown("Copia tu guión y descubre cómo hacerlo más viral 📈")

script_input = st.text_area("✍️ Pega aquí tu guión", height=200)

if st.button("📊 Analizar Guión"):
    if not script_input.strip():
        st.warning("⚠️ Debes ingresar un guión para analizar.")
    else:
        st.subheader("📋 Análisis del Script")
        sentimiento = detectar_sentimiento(script_input)
        viralidad = evaluar_viralidad(script_input)
        recomendaciones = generar_recomendaciones(script_input)

        st.markdown(f"**🔍 Sentimiento detectado:** {sentimiento.capitalize()}")
        st.markdown(f"**🔥 Potencial de viralidad:** {viralidad}")

        st.markdown("**💡 Recomendaciones:**")
        for rec in recomendaciones:
            st.markdown(f"- {rec}")

        mejorado = mejorar_script(script_input)
        st.subheader("📝 Versión Mejorada del Guión")
        st.text(mejorado)
