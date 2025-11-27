import asyncio
import os
from typing import Annotated

from pydantic import BaseModel, Field

from openai import OpenAI

from agents import Agent, Runner, function_tool, set_default_openai_key


OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
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
    """Analiza el texto de la conversación y devuelve la palabra más repetida."""
    import re
    from collections import Counter

    print("[debug] Analizando conversación...")

    # Limpiar texto y convertir a minúsculas
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
    print(f"regresando los {len(response.output)} documentos encontrados")
    

    messages = [
        m
        for m in response.output
        if m.type == "message"
    ]

    if not messages:
        return "-- No response --"

    assistance_message_text = messages[0].content[0].text
    print(assistance_message_text)
 
    return RagSimilarity(text=assistance_message_text)
    

async def main():
    print("🚀 Agentes con OpenAI Agents SDK")

    agent = Agent(
        name="Agente de atención a cliente Universidad Panamericana",
        instructions="Responde como un asistente que aclara dudas de la universida panamericana.",
        tools=[get_word, rag_funtion]
    )

    conversation = client.conversations.create()

   
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Adiós!")
            break

        result = await Runner.run(agent, input=user_input, conversation_id=conversation.id)

        print("🟩 Respuesta del agente:")
        print(result.final_output)
       
    print("✅ Conversación terminada.")
    

    



if __name__ == "__main__":
    asyncio.run(main())
