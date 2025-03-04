import datetime
import chromadb
import traceback
import os  # 新增 os 模組來檢查檔案是否存在

import pandas as pd
from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() == 0:
        try:
            df = pd.read_csv("COA_OpenData.csv")

            for idx, row in df.iterrows():
                metadata = {
                    "file_name": "COA_OpenData.csv",
                    "name": row['Name'],
                    "type": row['Type'],
                    "address": row['Address'],
                    "tel": row['Tel'],
                    "city": row['City'],
                    "town": row['Town'],
                    "date": int(datetime.datetime.strptime(row['CreateDate'], "%Y-%m-%d").timestamp())
                }
                collection.add(
                    ids=[str(idx)],
                    documents=[row['HostWords']],
                    metadatas=[metadata]
                )
                print(f"Successfully added document with id: {str(idx)} and metadata: {metadata}")
        except Exception as e:
            print(f"Error occurred: {e}")
            raise
    else:
        print("Collection 'TRAVEL' already contains data, skipping generation.")
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    where_conditions = []
    if city:
        where_conditions.append({"city": {"$in": city}})
    if store_type:
        where_conditions.append({"type": {"$in": store_type}})
    if start_date and end_date:
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        where_conditions.append({"date": {"$gte": start_timestamp}})
        where_conditions.append({"date": {"$lte": end_timestamp}})

    where_filter = {"$and": where_conditions} if where_conditions else None

    results = collection.query(
        query_texts=[question],
        n_results=10,
        where=where_filter,
        include=["metadatas", "distances"]
    )

    filtered_results = []
    for i, distance in enumerate(results["distances"][0]):
        similarity = 1 - distance
        if similarity >= 0.80:
            filtered_results.append((results["metadatas"][0][i]["name"], similarity))
    
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    filtered_names = [name for name, _ in filtered_results]

    # 調試用：顯示相似度（不影響最終輸出格式）
    print(f"調試資訊：找到 {len(filtered_results)} 個符合條件的店家")
    for name, similarity in filtered_results:
        print(f"- {name}: 相似度 {similarity:.3f}")
    
    return filtered_names
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

if __name__ == "__main__":
    sqlite_path = os.path.join(dbpath, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        print(f"'{sqlite_path}' 已存在，跳過 generate_hw01() 的執行。")
    else:
        print(f"'{sqlite_path}' 不存在，開始執行 generate_hw01()。")
        try:
            collection = generate_hw01()
            print("hw01 執行完成，chroma.sqlite3 已生成。")
        except Exception as e:
            print(f"執行 generate_hw01() 時發生錯誤：{e}")
            traceback.print_exc()

    question = "我想要找有關茶餐點的店家"
    city = ["宜蘭縣", "新北市"]
    store_type = ["美食"]
    start_date = datetime.datetime(2024, 4, 1)
    end_date = datetime.datetime(2024, 5, 1)

    result = generate_hw02(question, city, store_type, start_date, end_date)
    print(result)  # 符合題目要求的格式