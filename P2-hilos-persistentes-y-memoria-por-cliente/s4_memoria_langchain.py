"""Sesión 4. Memoria persistente con LangChain: hilos (corto plazo) y preferencias (largo plazo).

VERSIÓN ENTREGABLE E4: incluye olvido completo (preferencias + hilos del cliente).

Uso:  python s4_memoria_langchain.py <cliente_id> <hilo>
      p. ej. "python s4_memoria_langchain.py 1 lunes" y después "... 1 martes".

Adaptado para funcionar con Ollama local via langchain-openai (ChatOpenAI apuntando a Ollama)."""
import os
import sqlite3
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

load_dotenv()

# Modelo: ChatOpenAI apuntando a Ollama (compatible con OpenAI API)
OLLAMA_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
MODELO_NOMBRE = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

llm = ChatOpenAI(
    model=MODELO_NOMBRE,
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    temperature=0,
)


@dataclass
class Contexto:
    cliente_id: str  # lo fija la aplicación tras identificar al cliente; el modelo no puede cambiarlo


def _espacio(runtime: ToolRuntime[Contexto]) -> tuple[str, str]:
    return ("clientes", runtime.context.cliente_id)  # namespace: aísla la memoria de cada cliente


OLVIDO_PENDIENTE: set[str] = set()


def borrar_hilos_cliente(cliente_id: str, excepto: str | None = None) -> int:
    """Borra del checkpointer los hilos cuyo thread_id es '<cliente_id>-<hilo>'.
    SqliteSaver no tiene list_threads: se listan con una consulta a su tabla."""
    filas = checkpointer.conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ?",
        (f"{cliente_id}-%",)).fetchall()
    n = 0
    for (thread_id,) in filas:
        if thread_id != excepto:
            checkpointer.delete_thread(thread_id)
            n += 1
    return n


@tool
def recordar_preferencia(clave: str, valor: str, runtime: ToolRuntime[Contexto]) -> str:
    """Guarda una preferencia estable del cliente (horario de entrega, producto habitual,
    forma de contacto). No guardes datos personales sensibles ni datos de pago."""
    runtime.store.put(_espacio(runtime), clave, {"valor": valor})
    return f"Preferencia '{clave}' guardada."


@tool
def ver_preferencias(runtime: ToolRuntime[Contexto]) -> dict:
    """Devuelve las preferencias guardadas del cliente actual."""
    return {item.key: item.value["valor"] for item in runtime.store.search(_espacio(runtime))}


@tool
def olvidar_preferencias(runtime: ToolRuntime[Contexto]) -> str:
    """Borra todas las preferencias del cliente actual Y todos sus hilos de conversación.
    Úsala si el cliente solicita el olvido completo de sus datos."""
    cliente_id = runtime.context.cliente_id

    # 1. Borrar preferencias del store (memoria a largo plazo)
    espacio = _espacio(runtime)
    items_borrados = 0
    for item in runtime.store.search(espacio):
        runtime.store.delete(espacio, item.key)
        items_borrados += 1

    # 2. Borrar los hilos del cliente (memoria a corto plazo). El hilo actual está en uso:
    # el checkpointer lo vuelve a escribir al terminar este turno, así que se marca como
    # pendiente y lo borra la aplicación cuando el turno ha terminado (ver main).
    hilos_borrados = borrar_hilos_cliente(cliente_id, excepto=runtime.config["configurable"]["thread_id"])
    OLVIDO_PENDIENTE.add(cliente_id)
    return (f"Olvido completo: {items_borrados} preferencias y {hilos_borrados} hilo(s) "
            f"del cliente {cliente_id} eliminados (la conversación actual se borra al cerrar el turno).")


SISTEMA = """Eres el asistente comercial de Tostadero Retiro.
Al empezar cada conversación consulta ver_preferencias y tenlas en cuenta.
Cuando el cliente exprese una preferencia estable, guárdala con recordar_preferencia.
Si pide que olvides sus datos, usa olvidar_preferencias y confírmalo."""

# Dos almacenes distintos: hilos de conversación y memoria a largo plazo
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_HILOS_DB = os.path.join(_DB_DIR, "hilos.db")
_MEMORIA_DB = os.path.join(_DB_DIR, "memoria.db")

checkpointer = SqliteSaver(sqlite3.connect(_HILOS_DB, check_same_thread=False))
store = SqliteStore(sqlite3.connect(_MEMORIA_DB, check_same_thread=False, isolation_level=None))
store.setup()

agente = create_agent(
    model=llm,
    tools=[recordar_preferencia, ver_preferencias, olvidar_preferencias],
    system_prompt=SISTEMA,
    context_schema=Contexto,
    checkpointer=checkpointer,
    store=store,
    middleware=[SummarizationMiddleware(model=llm, trigger=("tokens", 3000),
                                        keep=("messages", 10))],
)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python s4_memoria_langchain.py <cliente_id> <hilo>")
        sys.exit(1)
    cliente_id, hilo = sys.argv[1], sys.argv[2]
    config = {"configurable": {"thread_id": f"{cliente_id}-{hilo}"}}
    print(f"[Cliente: {cliente_id} | Hilo: {hilo}]")
    while texto := input("Cliente: ").strip():
        estado = agente.invoke({"messages": [{"role": "user", "content": texto}]},
                               config, context=Contexto(cliente_id=cliente_id))
        print("Asistente:", estado["messages"][-1].text)
        if cliente_id in OLVIDO_PENDIENTE:  # turno terminado: ya se puede borrar el hilo actual
            borrar_hilos_cliente(cliente_id)
            OLVIDO_PENDIENTE.discard(cliente_id)
            print("   (hilos del cliente eliminados; fin de la sesión)")
            break
