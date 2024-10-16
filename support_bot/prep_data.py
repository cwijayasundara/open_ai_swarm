import json
import os
import pandas as pd
import qdrant_client
from openai import OpenAI
from qdrant_client.http import models as rest
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
_ = load_dotenv()

client = OpenAI()

GPT_MODEL = "gpt-4o-mini"

EMBEDDING_MODEL = "text-embedding-3-small"

article_list = os.listdir("data")

articles = []

for x in article_list:
    article_path = "data/" + x
    f = open(article_path)
    data = json.load(f)
    articles.append(data)
    f.close()

for i, x in enumerate(articles):
    try:
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=x["text"])
        articles[i].update({"embedding": embedding.data[0].embedding})
    except Exception as e:
        print(x["title"])
        print(e)

qdrant = qdrant_client.QdrantClient(host="localhost", port=6335)

print("qdrant collections", qdrant.get_collections())

collection_name = "help_center"

vector_size = len(articles[0]["embedding"])
print("vector size", vector_size)

article_df = pd.DataFrame(articles)
print(article_df.head())

# Delete the collection if it exists, so we can rewrite it changes to articles were made
collections = qdrant.get_collections().collections
if any(col.name == collection_name for col in collections):
    qdrant.delete_collection(collection_name=collection_name)

# Create Vector DB collection
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config={
        "article": rest.VectorParams(
            distance=rest.Distance.COSINE,
            size=vector_size,
        )
    },
)

# Populate collection with vectors
qdrant.upsert(
    collection_name=collection_name,
    points=[
        rest.PointStruct(
            id=k,
            vector=v["embedding"],
            payload=v.to_dict(),
        )
        for k, v in article_df.iterrows()
    ],
)