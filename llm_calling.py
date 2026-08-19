from embeddings import query_embedding
from Faiss_Searach import query_indices
from embeddings import embeddings_list_whole
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()



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
def the_call(query):
    query_vector = query_embedding(query)
    scores, indices = query_indices(query_vector, top_k=3)
    # return scores, indices
    relivant_chunks = []
    for i in range(len(indices[0])):
        index = indices[0][i]
        relivant_chunks.append(embeddings_list_whole[index])
        file, content, embeddings = relivant_chunks[-1]
        # print(f"File: {file}")
        # print(f"Content: {content}")
        # # print(f"Embeddings: {embeddings}")
        # print("-" * 40)

    context_text = "\n\n".join([chunk[1] for chunk in relivant_chunks])
    print(generate_answer(query, context=context_text))

the_call(query)