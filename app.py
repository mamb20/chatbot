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

# ------------------------
# BARRA LATERAL
# ------------------------
with st.sidebar:
    st.image("https://apseguridad.com/wp-content/uploads/2007/01/escudo_CH001.jpg", width=150)

    st.title("Menú lateral")
    

   if st.session_state.modo_oscuro:
    # ---------- MODO OSCURO ----------
    st.markdown("""
    <style>
        .stApp { background-color: #0f0f0f !important; color: white !important; }
        .chat-box { background-color: #1c1c1c !important; border: 1px solid #333 !important; }
        .user-msg { background-color: #2a4d29 !important; color: white !important; }
        .bot-msg { background-color: #262626 !important; border: 1px solid #555 !important; color: #e8e8e8 !important; }
        input[type="text"] { background-color: #1c1c1c !important; color: white !important; border: 1px solid #444 !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    # ---------- MODO CLARO ----------
    st.markdown("""
    <style>
        .stApp { background-color: #f5f5f5 !important; color: black !important; }
        .chat-box { background-color: #ffffff !important; border-radius: 12px; border: 1px solid #eee !important; }
        .user-msg { background-color: #DCF8C6 !important; color: black !important; }
        .bot-msg { background-color: #ffffff !important; border: 1px solid #ddd !important; color: black !important; }
        input[type="text"] { background-color: white !important; color: black !important; border: 1px solid #ccc !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Información extra")
    st.write("Proyecto final de Programación orientada a objetos.")

    st.write("- Desarrollado por Miguel")

    st.markdown("---")


























