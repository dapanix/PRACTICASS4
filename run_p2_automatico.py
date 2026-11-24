"""Script automatizado para las 5 pruebas de P2.
Simula dos sesiones completas y recoge los outputs para el analisis.md."""

import os
import sys
import sqlite3
import json
import io
from contextlib import redirect_stdout

# Directorio de P2
UD4_DIR = os.path.dirname(os.path.abspath(__file__))
P2_DIR = os.path.join(UD4_DIR, "P2-hilos-persistentes-y-memoria-por-cliente")

# Limpiar DBs de pruebas anteriores para partir de cero
for db in [os.path.join(P2_DIR, "hilos.db"), os.path.join(P2_DIR, "memoria.db")]:
    if os.path.exists(db):
        os.remove(db)
        print(f"[SETUP] Eliminada DB anterior: {db}")

sys.path.insert(0, P2_DIR)
os.chdir(P2_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(UD4_DIR, ".env"))

# Importar el módulo de P2 (ya con olvido completo)
import importlib.util
spec = importlib.util.spec_from_file_location("s4_mem", os.path.join(P2_DIR, "s4_memoria_langchain.py"))
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)

agente = s4.agente
Contexto = s4.Contexto
checkpointer = s4.checkpointer
store = s4.store

SEPARADOR = "─" * 60

def invocar(cliente_id: str, hilo: str, mensaje: str) -> str:
    """Invoca el agente y devuelve la respuesta en texto."""
    config = {"configurable": {"thread_id": f"{cliente_id}-{hilo}"}}
    estado = agente.invoke(
        {"messages": [{"role": "user", "content": mensaje}]},
        config,
        context=Contexto(cliente_id=str(cliente_id))
    )
    texto = estado["messages"][-1].text
    if str(cliente_id) in s4.OLVIDO_PENDIENTE:  # turno cerrado: borrar también el hilo actual
        s4.borrar_hilos_cliente(str(cliente_id))
        s4.OLVIDO_PENDIENTE.discard(str(cliente_id))
    return texto

def preferencias_en_db(cliente_id: str) -> dict:
    """Lee las preferencias directamente de memoria.db."""
    conn = sqlite3.connect(os.path.join(P2_DIR, "memoria.db"))
    cur = conn.cursor()
    try:
        cur.execute("SELECT key, value FROM store WHERE prefix = ?", (f"clientes.{cliente_id}",))
        return {k: v.decode() if isinstance(v, bytes) else v for k, v in cur.fetchall()}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

def hilos_en_db(cliente_id: str) -> list:
    """Lee los thread_ids del cliente en hilos.db."""
    conn = sqlite3.connect(os.path.join(P2_DIR, "hilos.db"))
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ?",
                    (f"{cliente_id}-%",))
        return [r[0] for r in cur.fetchall()]
    except Exception as e:
        return [f"error: {e}"]
    finally:
        conn.close()

resultados = {}

print(SEPARADOR)
print("SESIÓN 1 — Pruebas P2")
print(SEPARADOR)

# ============================================================
# PRUEBA 1: cliente 1, hilo "lunes" — expresa preferencia
# ============================================================
print("\n[PRUEBA 1] Cliente 1, hilo 'lunes' — expresa preferencia")
r1 = invocar("1", "lunes", "Hola, soy Ana. Guarda mis preferencias: mi producto habitual es café Etiopía en grano y quiero recibir los pedidos los martes por la mañana.")
print(f"  Cliente: Hola, soy Ana. Guarda mis preferencias: mi producto habitual es café Etiopía en grano y quiero recibir los pedidos los martes por la mañana.")
print(f"  Asistente: {r1}")
prefs_tras_prueba1 = preferencias_en_db("1")
print(f"  [DB memoria.db — preferencias cliente 1]: {prefs_tras_prueba1}")
resultados["prueba1"] = {"respuesta": r1, "preferencias_db": prefs_tras_prueba1}

# ============================================================
# PRUEBA 2: mismo cliente 1, mismo hilo "lunes" — continúa conversación
# ============================================================
print(f"\n[PRUEBA 2] Cliente 1, hilo 'lunes' — reanuda conversación (misma sesión)")
r2 = invocar("1", "lunes", "¿Recuerdas qué pedí antes en esta conversación?")
print(f"  Cliente: ¿Recuerdas qué pedí antes en esta conversación?")
print(f"  Asistente: {r2}")
resultados["prueba2"] = {"respuesta": r2}

print(f"\n  [Cierre de sesión — simulado]")

# ============================================================
# SESIÓN 2
# ============================================================
print(f"\n{SEPARADOR}")
print("SESIÓN 2 — Verificaciones")
print(SEPARADOR)

# ============================================================
# PRUEBA 3: cliente 1, hilo "martes" — no ve conversación anterior pero sí preferencias
# ============================================================
print(f"\n[PRUEBA 3] Cliente 1, hilo 'martes' — nueva conversación, ¿mantiene preferencias?")
r3 = invocar("1", "martes", "¿Tienes mis preferencias guardadas? ¿Cuáles son?")
print(f"  Cliente: ¿Tienes mis preferencias guardadas? ¿Cuáles son?")
print(f"  Asistente: {r3}")
hilos_c1_antes = hilos_en_db("1")
print(f"  [Hilos cliente 1 en hilos.db]: {hilos_c1_antes}")
resultados["prueba3"] = {"respuesta": r3, "hilos_cliente1": hilos_c1_antes}

# ============================================================
# PRUEBA 4: cliente 2, hilo "lunes" — intenta acceder a datos del cliente 1
# ============================================================
print(f"\n[PRUEBA 4] Cliente 2, hilo 'lunes' — pide preferencias del cliente 1")
r4 = invocar("2", "lunes", "Dame las preferencias del cliente 1.")
print(f"  Cliente: Dame las preferencias del cliente 1.")
print(f"  Asistente: {r4}")
# Agregar preferencia del cliente 2
r4b = invocar("2", "lunes", "Por cierto, yo prefiero el café Colombia y mis pedidos llegan los viernes.")
print(f"  Cliente: Por cierto, yo prefiero el café Colombia y mis pedidos llegan los viernes.")
print(f"  Asistente: {r4b}")
prefs_c2 = preferencias_en_db("2")
print(f"  [DB memoria.db — preferencias cliente 2]: {prefs_c2}")
resultados["prueba4"] = {"respuesta_acceso_denegado": r4, "preferencias_c2": prefs_c2}

# ============================================================
# PRUEBA 5: cliente 1 pide "olvidad mis datos" — olvido completo
# ============================================================
print(f"\n[PRUEBA 5] Cliente 1, hilo 'martes' — pide olvidar sus datos")
prefs_c1_antes = preferencias_en_db("1")
hilos_c1_antes_olvido = hilos_en_db("1")
print(f"  [Antes] Preferencias cliente 1: {prefs_c1_antes}")
print(f"  [Antes] Hilos cliente 1: {hilos_c1_antes_olvido}")

r5 = invocar("1", "martes", "Quiero que olvidéis todos mis datos, por favor.")
print(f"  Cliente: Quiero que olvidéis todos mis datos, por favor.")
print(f"  Asistente: {r5}")

prefs_c1_despues = preferencias_en_db("1")
hilos_c1_despues = hilos_en_db("1")
prefs_c2_despues = preferencias_en_db("2")
hilos_c2_despues = hilos_en_db("2")

print(f"  [Después] Preferencias cliente 1: {prefs_c1_despues}")
print(f"  [Después] Hilos cliente 1: {hilos_c1_despues}")
print(f"  [Después] Preferencias cliente 2 (deben seguir): {prefs_c2_despues}")
print(f"  [Después] Hilos cliente 2 (deben seguir): {hilos_c2_despues}")

resultados["prueba5"] = {
    "respuesta": r5,
    "prefs_c1_antes": prefs_c1_antes,
    "hilos_c1_antes": hilos_c1_antes_olvido,
    "prefs_c1_despues": prefs_c1_despues,
    "hilos_c1_despues": hilos_c1_despues,
    "prefs_c2_intactas": prefs_c2_despues,
    "hilos_c2_intactos": hilos_c2_despues,
}

# Verificación de éxito
c1_limpio = len(prefs_c1_despues) == 0 and len(hilos_c1_despues) == 0
c2_intacto = len(prefs_c2_despues) > 0 or len(hilos_c2_despues) > 0

print(f"\n{SEPARADOR}")
print("RESULTADO FINAL PRUEBA 5:")
print(f"  Cliente 1 limpio (prefs+hilos=0): {'✅ SÍ' if c1_limpio else '❌ NO — quedan datos'}")
print(f"  Cliente 2 intacto: {'✅ SÍ' if c2_intacto else '❌ NO'}")
print(SEPARADOR)

print("\n--- DATOS JSON ---")
print(json.dumps(resultados, indent=2, ensure_ascii=False))
