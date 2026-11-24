"""Script automatizado para P1: ejecuta 12 turnos de conversación simulada
y recoge los datos de tokens antes/después de compactar.
Adapta s4_compactar.py para usar Ollama via OpenAI SDK."""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Cargar .env del directorio de trabajo
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
MODELO = "llama3.2:latest"
SISTEMA = "Eres el asistente comercial de Tostadero Retiro. Responde en español, con brevedad."
UMBRAL_TOKENS_ORIGINAL = 1000
MANTENER = 6  # par, como exige el diseño

resultados = []  # para el análisis

def compactar(mensajes: list[dict], umbral_label: str = "original") -> list[dict]:
    antiguos = mensajes[:-MANTENER]
    recientes = mensajes[-MANTENER:]
    transcripcion = "\n".join(f"{m['role']}: {m['content']}" for m in antiguos)
    r = client.chat.completions.create(
        model=MODELO,
        max_tokens=600,
        messages=[
            {"role": "system", "content": "Resume esta conversación de atención al cliente. Conserva datos concretos: nombres, ids, cantidades, fechas, compromisos adquiridos y temas pendientes."},
            {"role": "user", "content": transcripcion},
        ],
    )
    resumen = r.choices[0].message.content
    return [
        {"role": "user", "content": f"[Resumen de la conversación hasta aquí]\n{resumen}"},
        {"role": "assistant", "content": "Entendido. Continúo con ese contexto."},
        *recientes,  # empieza por mensaje de usuario porque MANTENER es par
    ]


def ejecutar_conversacion(umbral_tokens: int, mantener: int, turnos_dialogo: list[str], etiqueta: str):
    """Ejecuta una conversación completa y devuelve los datos de tokens."""
    global MANTENER
    MANTENER = mantener
    historial: list[dict] = []
    tokens_historial = 0
    datos = []
    compactaciones = []
    tokens_antes_compactar = None

    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {etiqueta} (UMBRAL={umbral_tokens}, MANTENER={mantener})")
    print(f"{'='*60}")

    for i, texto in enumerate(turnos_dialogo, 1):
        compactado_este_turno = False
        if len(historial) > mantener and tokens_historial > umbral_tokens:
            tokens_antes_compactar = tokens_historial
            historial = compactar(historial)
            compactado_este_turno = True
            print(f"\n   *** COMPACTACIÓN en turno {i} (tokens antes: {tokens_antes_compactar}) ***")

        historial.append({"role": "user", "content": texto})

        messages = [{"role": "system", "content": SISTEMA}] + historial
        r = client.chat.completions.create(
            model=MODELO,
            max_tokens=200,
            messages=messages,
        )
        tokens_entrada = r.usage.prompt_tokens
        tokens_salida = r.usage.completion_tokens
        tokens_historial = tokens_entrada + tokens_salida
        respuesta = r.choices[0].message.content
        historial.append({"role": "assistant", "content": respuesta})

        print(f"T{i:02d} | Cliente: {texto[:60]}")
        print(f"     | Asistente: {respuesta[:80].replace(chr(10), ' ')}")
        print(f"     | Tokens entrada: {tokens_entrada} | Salida: {tokens_salida} | Total acum.: {tokens_historial}")
        if compactado_este_turno:
            compactaciones.append({
                "turno": i,
                "tokens_antes": tokens_antes_compactar,
                "tokens_despues": tokens_entrada
            })

        datos.append({
            "turno": i,
            "tokens_entrada": tokens_entrada,
            "tokens_salida": tokens_salida,
            "compactado": compactado_este_turno
        })

    return datos, compactaciones, historial


# ============================================================
# TURNO ESPECIAL: primera pregunta memorable
# ============================================================
PREGUNTA_TURNO_1 = "Hola, me llamo Carlos García y quiero pedir 5 kg de café Etiopía. Mi número de cliente es 7749."

DIALOGO = [
    PREGUNTA_TURNO_1,
    "¿Cuánto cuesta el café Etiopía por kilo?",
    "¿Tienen descuentos para pedidos grandes, de más de 10 kg?",
    "Perfecto. ¿Qué otros cafés de origen único tienen disponibles?",
    "¿El café de Guatemala tiene notas afrutadas?",
    "Me interesa también probar el café de Colombia. ¿Lo tienen en grano?",
    "¿Pueden enviar a Bilbao? ¿Cuánto cuesta el envío?",
    "¿El envío es gratuito si compro más de cierta cantidad?",
    "Entonces añado 2 kg de Colombia al pedido del Etiopía. Total 7 kg.",
    "¿Puedo pagar con transferencia bancaria?",
    "¿Cuántos días tarda en llegar a Bilbao el pedido?",
    "Perfecto, confirmo el pedido. Por favor, ¿cuál es el nombre del cliente que hizo este pedido?",
]

# ============================================================
# EXPERIMENTO 1: configuración original (UMBRAL=1000, MANTENER=6)
# ============================================================
datos1, compact1, hist1 = ejecutar_conversacion(
    umbral_tokens=1000,
    mantener=6,
    turnos_dialogo=DIALOGO,
    etiqueta="Config original (UMBRAL=1000, MANTENER=6)"
)

print("\n\n--- VERIFICACIÓN: ¿Sobrevivió el nombre/datos del turno 1? ---")
pregunta_memoria = "¿Recuerdas el nombre del cliente que hizo el primer pedido y su número de cliente?"
historial_test = hist1 + [{"role": "user", "content": pregunta_memoria}]
messages_test = [{"role": "system", "content": SISTEMA}] + historial_test
r_test = client.chat.completions.create(model=MODELO, max_tokens=150, messages=messages_test)
respuesta_memoria = r_test.choices[0].message.content
print(f"Pregunta: {pregunta_memoria}")
print(f"Respuesta: {respuesta_memoria}")

# ============================================================
# EXPERIMENTO 2: umbral más bajo (UMBRAL=400, MANTENER=4)
# ============================================================
datos2, compact2, hist2 = ejecutar_conversacion(
    umbral_tokens=400,
    mantener=4,
    turnos_dialogo=DIALOGO,
    etiqueta="Config alternativa (UMBRAL=400, MANTENER=4)"
)

# ============================================================
# EXPERIMENTO 3: umbral muy alto (UMBRAL=5000, MANTENER=8)
# ============================================================
datos3, compact3, hist3 = ejecutar_conversacion(
    umbral_tokens=5000,
    mantener=8,
    turnos_dialogo=DIALOGO,
    etiqueta="Config alta (UMBRAL=5000, MANTENER=8)"
)

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n\n" + "="*60)
print("RESUMEN DE RESULTADOS")
print("="*60)

for label, datos, compact in [
    ("Config original (UMBRAL=1000, MANTENER=6)", datos1, compact1),
    ("Config alternativa (UMBRAL=400, MANTENER=4)", datos2, compact2),
    ("Config alta (UMBRAL=5000, MANTENER=8)", datos3, compact3),
]:
    print(f"\n{label}:")
    print(f"  Tokens entrada por turno: {[d['tokens_entrada'] for d in datos]}")
    print(f"  Compactaciones: {len(compact)}")
    for c in compact:
        print(f"    -> Turno {c['turno']}: {c['tokens_antes']} tokens antes -> {c['tokens_despues']} tokens después")
    if not compact:
        print("    -> No se compactó (umbral no alcanzado)")

print(f"\nVerificación de memoria (turno 1 -> turno 12):")
print(f"  {respuesta_memoria}")

print("\n\n--- DATOS JSON PARA analisis.md ---")
import json
print(json.dumps({
    "experimento1": {"datos": datos1, "compactaciones": compact1},
    "experimento2": {"datos": datos2, "compactaciones": compact2},
    "experimento3": {"datos": datos3, "compactaciones": compact3},
    "verificacion_memoria": respuesta_memoria,
}, indent=2, ensure_ascii=False))
