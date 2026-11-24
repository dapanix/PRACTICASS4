# E4 · Análisis: memoria y contexto con modelos locales

**Modelos:** todo se ha ejecutado en local con Ollama (`llama3.2:latest`, 3B) a través de su API compatible con OpenAI (`http://localhost:11434/v1`), en lugar de OpenRouter/Claude/Gemini. Sin claves ni coste por token. P1 usa el SDK de `openai`; P2 usa `langchain-openai` (`ChatOpenAI`).

Los registros completos están en [`registro_p1.log`](registro_p1.log) y [`registro_p2.log`](registro_p2.log). Se generaron con `run_p1_automatico.py` y `run_p2_automatico.py`, que simulan al cliente (no hay conversación manual).

---

## P1. Compactación desde cero

Conversación de 12 turnos (pedido de café de Carlos García, cliente 7749, 5 kg de Etiopía en el turno 1).

### Tokens de entrada antes y después de compactar (UMBRAL=1000, MANTENER=6)

Tokens de entrada por turno: `79, 156, 239, 330, 414, 491, 587, 711, 810, 925, 1006, 632`.

- El historial crece casi linealmente (~90 tokens por turno) y al terminar el turno 11 ocupa 1006 + salida = **1110 tokens**, por encima del umbral.
- En el turno 12 se compacta: **1110 tokens antes → 632 tokens de entrada después** (incluye el resumen, los 6 mensajes recientes y la pregunta nueva). Reducción de ~43 % en ese turno.

### Qué sobrevivió del turno 1

Tras compactar se preguntó: «¿Recuerdas el nombre del cliente que hizo el primer pedido y su número de cliente?». El modelo respondió **«Carlos García … número de cliente 7749»**: el resumen conservó nombre e id. Matiz: el prompt de resumen pide expresamente conservar «nombres, ids, cantidades…», y con un modelo de 3B no hay garantía de que lo haga siempre; solo se probó esta ejecución y la verificación se hizo únicamente en la configuración original.

### Efecto de cambiar `MANTENER` y `UMBRAL_TOKENS`

| Config | Compactaciones | Tokens de entrada por turno | Máximo |
| --- | --- | --- | --- |
| UMBRAL=1000, MANTENER=6 | 1 (turno 12) | 79, 156, 239, 330, 414, 491, 587, 711, 810, 925, 1006, 632 | 1006 |
| UMBRAL=400, MANTENER=4 | 7 (turnos 5–10 y 12) | 79, 190, 281, 398, 379, 361, 489, 501, 337, 342, 410, 257 | 501 |
| UMBRAL=5000, MANTENER=8 | 0 | 79, 177, 224, 322, 429, 471, 589, 651, 701, 809, 872, 926 | 926 |

- **Coste:** con umbral bajo el contexto se mantiene pequeño (máx. 501 frente a 1006), pero cada compactación es una llamada extra al modelo. Con 7 compactaciones en 12 turnos se hicieron 19 llamadas en lugar de 13; en una API de pago eso puede costar más que lo que se ahorra, y con un modelo local alarga el tiempo. Con umbral alto no hay llamadas extra, pero el contexto crece sin límite: a más turnos sería más caro y acabaría desbordando la ventana (4096 tokens de contexto por defecto en Ollama).
- **Calidad:** compactar tan seguido (UMBRAL=400) resume resúmenes: en el turno 7 la compactación incluso *aumentó* los tokens (427 → 489) porque el resumen más los mensajes recientes pesaba más que lo que se descartó. Cada pasada pierde detalle. Con umbral alto y sin compactar no se pierde nada, mientras cabe en la ventana. `MANTENER` mayor conserva más literal reciente (mejor coherencia inmediata) a costa de comprimir menos.
- Limitación: una sola ejecución por configuración con un modelo pequeño; las diferencias de calidad son cualitativas, no medidas.

### Por qué `MANTENER` debe ser par

El historial es una secuencia alternada `user, assistant, user, assistant…`. Tras compactar, el historial empieza con `[resumen]` (user) + «Entendido» (assistant) y le siguen los mensajes recientes. Esos recientes son los últimos `MANTENER` mensajes; como el historial termina siempre en `assistant` (respuesta ya añadida), un `MANTENER` par hace que el primero de ellos sea un `user`. Así se sigue alternando correctamente. Con un valor impar el primer mensaje reciente sería un `assistant` y se tendrían dos `assistant` seguidos, lo que la API de Anthropic rechaza (exige alternancia) y otros proveedores toleran peor.

---

## P2. Hilos persistentes y memoria por cliente

Dos almacenes: `hilos.db` (checkpointer `SqliteSaver`, corto plazo, un hilo por `thread_id = <cliente>-<hilo>`) y `memoria.db` (`SqliteStore`, largo plazo, espacio de nombres `("clientes", <cliente_id>)`).

### Registro de pruebas en dos sesiones

Salida completa en [`registro_p2.log`](registro_p2.log). Resumen:

**Sesión 1**

1. **Cliente 1, hilo `lunes`**, expresa preferencias → el modelo llama a `recordar_preferencia` y en `memoria.db` aparece `entrega_horario = "martes por la mañana"`. *(Sólo guardó una de las dos preferencias: el producto habitual no se guardó.)*
2. **Cliente 1, hilo `lunes`** (continúa) → «¿Recuerdas qué pedí…?» responde que el producto habitual es el café Etiopía en grano: recupera la conversación del checkpointer.

*(Se cierra el script: la sesión 2 es otra invocación con las bases de datos ya escritas en disco.)*

**Sesión 2**

3. **Cliente 1, hilo `martes`** (conversación nueva) → «¿Tienes mis preferencias guardadas?» responde que sí y menciona el martes por la mañana: ve las preferencias (store) pero no la conversación del lunes. *(La redacción de la respuesta es confusa, típica del modelo pequeño.)* Hilos del cliente 1 en `hilos.db`: `['1-lunes', '1-martes']`.
4. **Cliente 2, hilo `lunes`** → «Dame las preferencias del cliente 1.» responde que no tiene esa información: el espacio de nombres lo fija la aplicación (`Contexto.cliente_id`), no el modelo. El cliente 2 guarda las suyas: `producto habitual = cafe colombia`, `horario de entrega = viernes` (y una `forma de contacto = "No especificado"` inventada por el modelo).
5. **Cliente 1, hilo `martes`** → «Quiero que olvidéis todos mis datos»:

| | Antes | Después |
| --- | --- | --- |
| Preferencias cliente 1 (`memoria.db`) | `entrega_horario` | *(vacío)* |
| Hilos cliente 1 (`hilos.db`) | `1-lunes`, `1-martes` | *(vacío)* |
| Preferencias cliente 2 | 3 preferencias | 3 preferencias (intactas) |
| Hilos cliente 2 | `2-lunes` | `2-lunes` (intacto) |

### Olvido completo: implementación

`olvidar_preferencias` borra las preferencias del store y los hilos del cliente con `checkpointer.delete_thread(thread_id)`. Dos detalles que no estaban en el enunciado:

- `SqliteSaver` **no tiene `list_threads`**, así que los hilos del cliente se listan con una consulta SQL a la tabla `checkpoints` (`thread_id LIKE '<cliente>-%'`).
- **¿Por qué sobrevive el hilo desde el que se pide el olvido?** Porque está en ejecución: cuando la herramienta lo borra, el agente aún tiene que producir la respuesta final y el checkpointer **vuelve a escribir** ese hilo (con la petición de olvido y la respuesta) al terminar el turno. **Solución:** la herramienta borra todos los demás hilos y marca al cliente como «olvido pendiente»; la aplicación, **cuando el turno ha terminado**, borra también el hilo actual (y en modo interactivo cierra la sesión). Resultado: no queda ningún hilo del cliente 1.

### Limitaciones observadas con el modelo local

- `llama3.2` (3B) llama a las herramientas de forma poco fiable. En una primera ejecución con un mensaje menos explícito («Siempre pido café Etiopía…») **no guardó ninguna preferencia** y respondió usando solo el contexto de la conversación; hubo que pedir explícitamente «Guarda mis preferencias…». También inventa valores («No especificado»). Con modelos mayores esto mejora; es la razón de ser de P3 (inyectar las preferencias en el prompt en vez de depender de que el modelo las consulte).
- La integridad del olvido **no depende del modelo**: la parte crítica (borrar hilos) se hace en código determinista. Solo depende de que el modelo decida llamar a la herramienta; en las pruebas lo hizo.

---

## Privacidad: qué guarda el asistente

| Dato | Dónde | Para qué | Cuánto tiempo | Cómo se borra |
| --- | --- | --- | --- | --- |
| Mensajes de cada conversación (texto del cliente y respuestas, incluido lo que el cliente cuente: nombre, pedidos…) | `hilos.db`, tabla `checkpoints`/`writes` (SQLite, checkpointer de LangGraph), un hilo por `<cliente>-<hilo>` | Continuar la conversación entre sesiones (memoria a corto plazo) | Indefinido: no hay caducidad automática. *Mejora recomendada:* política de retención (p. ej. 30 días) | Al pedir el olvido: `delete_thread` de todos los hilos del cliente |
| Preferencias estables (horario de entrega, producto habitual, contacto) | `memoria.db`, tabla `store`, `prefix = clientes.<id>` | Personalizar futuras conversaciones (memoria a largo plazo) | Indefinido hasta que el cliente pida el olvido | Al pedir el olvido: se borran todas las claves del espacio del cliente |

- **Alcance:** todo se guarda **en local**, en ficheros SQLite de la máquina que ejecuta el asistente. Al usar modelos locales con Ollama, **ninguna conversación sale al exterior**: ni a OpenRouter ni a ningún proveedor de IA. Esto es una ventaja de privacidad frente a la configuración original del curso.
- **Aislamiento entre clientes:** el `cliente_id` lo fija la aplicación en el contexto de ejecución; el modelo no puede cambiarlo, por lo que no puede leer ni borrar datos de otro cliente (prueba 4).
- **Minimización:** la herramienta indica al modelo que no guarde datos sensibles ni de pago. Es solo una instrucción al modelo y no una garantía: el texto íntegro de la conversación sí queda en `hilos.db`. Un modelo pequeño puede además guardar valores inventados.
- **Ejercicio del derecho de supresión:** el cliente lo solicita en lenguaje natural («olvidad mis datos»); el asistente llama a `olvidar_preferencias`, que borra preferencias y todos los hilos (incluido el actual al cerrar el turno). Se verifica abriendo ambas bases de datos (ver tabla de la prueba 5).
- **Pendiente / riesgos:** SQLite sin cifrar (conviene cifrar el disco o la base de datos), sin autenticación real del cliente (en producción el `cliente_id` debe salir de un login), sin retención automática y sin registro de auditoría de los borrados. Los ficheros `*.db` se excluyen del repositorio.

## P3 y P4

No forman parte del entregable. P3 (opcional) no se ha implementado. P4 es una demo con Gemini nativo, que requiere `GEMINI_API_KEY` y almacena el estado en los servidores de Google; no es compatible con el enfoque 100 % local, así que no se ha ejecutado.
