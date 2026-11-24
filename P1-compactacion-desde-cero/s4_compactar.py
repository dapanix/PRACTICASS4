"""Sesión 4. Memoria de corto plazo desde cero: compactar el historial cuando crece.

Versión con modelo LOCAL: usa el SDK de OpenAI contra Ollama (http://localhost:11434/v1)
en lugar del SDK de Anthropic contra OpenRouter. La lógica de compactación es la misma."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
llm = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
             api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
MODELO = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
SISTEMA = "Eres el asistente comercial de Tostadero Retiro. Responde en español, con brevedad."
UMBRAL_TOKENS = 1000  # bajo a propósito, para ver la compactación en clase
MANTENER = 6          # mensajes recientes que se conservan literales (número par)


def compactar(mensajes: list[dict]) -> list[dict]:
    antiguos, recientes = mensajes[:-MANTENER], mensajes[-MANTENER:]
    transcripcion = "\n".join(f"{m['role']}: {m['content']}" for m in antiguos)
    r = llm.chat.completions.create(
        model=MODELO, max_tokens=600,
        messages=[
            {"role": "system", "content": (
                "Resume esta conversación de atención al cliente. Conserva datos concretos: "
                "nombres, ids, cantidades, fechas, compromisos adquiridos y temas pendientes.")},
            {"role": "user", "content": transcripcion},
        ],
    )
    resumen = r.choices[0].message.content
    return [
        {"role": "user", "content": f"[Resumen de la conversación hasta aquí]\n{resumen}"},
        {"role": "assistant", "content": "Entendido. Continúo con ese contexto."},
        *recientes,  # empieza por un mensaje de usuario porque MANTENER es par
    ]


if __name__ == "__main__":
    historial: list[dict] = []
    tokens_historial = 0  # tamaño del historial según el uso que devolvió la última llamada
    while texto := input("Cliente: ").strip():
        if len(historial) > MANTENER and tokens_historial > UMBRAL_TOKENS:
            historial = compactar(historial)
            print("   (historial compactado)")
        historial.append({"role": "user", "content": texto})
        r = llm.chat.completions.create(
            model=MODELO, max_tokens=800,
            messages=[{"role": "system", "content": SISTEMA}, *historial])
        tokens_historial = r.usage.prompt_tokens + r.usage.completion_tokens
        respuesta = r.choices[0].message.content
        historial.append({"role": "assistant", "content": respuesta})
        print(f"Asistente: {respuesta}\n   [{r.usage.prompt_tokens} tokens de entrada en esta llamada]")
