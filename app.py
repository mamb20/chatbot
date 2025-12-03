import streamlit as st
import asyncio
import os
from typing import Annotated

from pydantic import BaseModel, Field
from openai import OpenAI

from agents import Agent, Runner, function_tool, set_default_openai_key




OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = "vs_6913baba995c81918b7f38c033955571"

set_default_openai_key(OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)





class WordAnalysis(BaseModel):
    word: str = Field(description="La palabra más repetida en la conversación")
    count: int = Field(description="Número de veces que aparece la palabra")


class RagSimilarity(BaseModel):
    text: str = Field(description="resultados de la busqueda de la universida panamericana")


@function_tool
def get_word(
    conversation_text: Annotated[str, "Texto completo de la conversación hasta el momento"]
) -> WordAnalysis:
    import re
    from collections import Counter

    print("[debug] Analizando conversación...")

    clean_text = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9 ]", "", conversation_text.lower())
    words = clean_text.split()

    if not words:
        return WordAnalysis(word="", count=0)

    counter = Counter(words)
    word, count = counter.most_common(1)[0]
    return WordAnalysis(word=word, count=count)


@function_tool
def rag_funtion(
    query: Annotated[str, "busca todo lo relacionado sobre historia u oferta acádemica de la universidad panamerica"]
) -> RagSimilarity:

    print("[debug] buscando en el RAG...")

    response = client.responses.create(
        model="gpt-5-mini",
        input=query,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "max_num_results": 2
        }],
    )

    messages = [m for m in response.output if m.type == "message"]

    if not messages:
        return RagSimilarity(text="-- No response --")

    assistance_message_text = messages[0].content[0].text
    return RagSimilarity(text=assistance_message_text)


agent = Agent(
    name="Agente de atención a cliente Universidad Panamericana",
    instructions="Responde como un asistente que aclara dudas de la universida panamericana.",
    tools=[get_word, rag_funtion]
)






st.title(" Chat Universidad Panamericana")


# Conversación de OpenAI (persistente)
if "conversation_id" not in st.session_state:
    conv = client.conversations.create()
    st.session_state.conversation_id = conv.id

# Historial visual
if "messages" not in st.session_state:
    st.session_state.messages = []


# Mostrar historial en formato tipo chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



user_input = st.chat_input("inicia conversación con Ubot")



if user_input:

    # Guardar mensaje del usuario en historial
    st.session_state.messages.append({"role": "user", "content": user_input})
   
    with st.chat_message("user"):
        st.write(user_input)

  
    async def run_agent():
        return await Runner.run(
            agent,
            input=user_input,
            conversation_id=st.session_state.conversation_id
        )
       

    result = asyncio.run(run_agent())
    respuesta = result.final_output
   
  
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
  
    with st.chat_message("assistant"):
        
        st.write(respuesta)


import streamlit as st

# -----------------------
# 1. Variable de selección
# -----------------------
if "seleccion" not in st.session_state:
    st.session_state.seleccion = "Inicio"

# -----------------------
# 2. Barra lateral (menú)
# -----------------------
st.sidebar.title("Menú")

if st.sidebar.button("🏠 Inicio"):
    st.session_state.seleccion = "Inicio"

if st.sidebar.button("💬 Chat"):
    st.session_state.seleccion = "Chat"

if st.sidebar.button("⚙️ Configuración"):
    st.session_state.seleccion = "Configuración"

if st.sidebar.button("ℹ️ Acerca de"):
    st.session_state.seleccion = "Acerca de"

# -----------------------
# 3. Contenido según menú
# -----------------------
if st.session_state.seleccion == "Inicio":
    st.title("🏠 Inicio")
    st.write("Bienvenido a la página principal.")

elif st.session_state.seleccion == "Chat":
    st.title("💬 Chat")
    st.write("Aquí iría tu chatbot.")

elif st.session_state.seleccion == "Configuración":
    st.title("⚙️ Configuración")
    st.write("Aquí puedes poner ajustes de tu app.")

elif st.session_state.seleccion == "Acerca de":
    st.title("ℹ️ Acerca de")
    st.write("Información de la aplicación.")



  





























