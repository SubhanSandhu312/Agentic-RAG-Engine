from Faiss_Searach import indices
from embeddings import embeddings_list_whole
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

relivant_chunks = []
for i in range(len(indices[0])):
    index = indices[0][i]
    relivant_chunks.append(embeddings_list_whole[index])
    file, content, embeddings = relivant_chunks[-1]
    print(f"File: {file}")
    print(f"Content: {content}")
    # print(f"Embeddings: {embeddings}")
    print("-" * 40)





load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def generate_answer(query, context):

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


query = "What is the issue in the docker-build-fail.yml file and how can it be resolved?"

print(generate_answer(query, context="\n".join([content for _, content, _ in relivant_chunks])))