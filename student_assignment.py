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
    # 嘗試刪除現有的 "TRAVEL" collection
    try:
        chroma_client.delete_collection(name="TRAVEL")
        print("Successfully deleted existing collection 'TRAVEL'")
    except ValueError as e:
        print(f"Collection 'TRAVEL' does not exist, skipping delete. {e}")
        
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    # 讀取 CSV 檔案
    df = pd.read_csv("COA_OpenData.csv")

    # 將資料寫入 collection
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
    
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
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

# 主程式邏輯：檢查 chroma.sqlite3 是否存在
if __name__ == "__main__":
    sqlite_path = os.path.join(dbpath, "chroma.sqlite3")  # 構建 chroma.sqlite3 的完整路徑
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