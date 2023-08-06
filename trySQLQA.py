import pandas as pd
import torch
from elasticsearch import Elasticsearch
from transformers import AutoModel,AutoTokenizer
import tqdm

# 初始化Elasticsearch
es = Elasticsearch(hosts=["http://localhost:9200"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode_text(bert_model, bert_tokenizer, text):
    # encoding
    inputs = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

def get_bert_embedding(myV):
    # embedding
    bert_model = AutoModel.from_pretrained("./bert-base-chinese")
    bert_tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese")
    query_vector=encode_text(bert_model, bert_tokenizer, myV)
    del bert_model
    del bert_tokenizer
    
    return query_vector.flatten().tolist()

def insert_into_es(df,es,index_name,id_col_name):
    mapping = {
        "mappings": {
            "properties": {
                "col_name": {"type": "keyword"},
                "row_i": {"type": "integer"},
                "id_col_name": {"type": "keyword"},
                "id_col_name_val": {"type": "keyword"},
                "value": {"type": "keyword"},
                "col_name_vec": {"type": "dense_vector", "dims": 768},  # 根据向量维度进行调整
                "value_vec": {"type": "dense_vector", "dims": 768}  # 根据向量维度进行调整
            }
        }
    }
    # 检查索引是否存在
    index_exists = es.indices.exists(index=index_name)
    # 如果索引不存在，则创建新索引
    if not index_exists:
        es.indices.create(index=index_name,body=mapping)
        print(f"索引 '{index_name}' 创建成功")
    else:
        print(f"索引 '{index_name}' 已存在")
    
    for col in tqdm.tqdm(df.columns):
        for i, cell in enumerate(df[col].values):
            if pd.notna(cell) and isinstance(cell, str): # 值输入
                # 拆分数据
                col_name = col
                row_i = i
                id_col_name_val = df[id_col_name][i] if pd.notna(df[id_col_name][i]) else ""
                value = cell

                # 使用BERT嵌入向量化（伪代码）
                col_name_vec = get_bert_embedding(col_name)
                value_vec = get_bert_embedding(value)

                # 构造文档数据
                doc = {
                    "col_name": col_name,
                    "row_i": row_i,
                    "id_col_name": id_col_name,
                    "id_col_name_val": id_col_name_val,
                    "value": value,
                    "col_name_vec": col_name_vec,
                    "value_vec": value_vec
                }

                # 将文档数据插入Elasticsearch索引
                es.index(index=index_name, doc_type='_doc', body=doc)
            else: # 列输入
                # 拆分数据
                col_name = col
                row_i = i
                id_col_name_val = df[id_col_name][i] if pd.notna(df[id_col_name][i]) else ""

                # 使用BERT嵌入向量化（伪代码）
                col_name_vec = get_bert_embedding(col_name)
                value_vec = get_bert_embedding(value)

                # 构造文档数据
                doc = {
                    "col_name": col_name,
                    "row_i": row_i,
                    "id_col_name": id_col_name,
                    "id_col_name_val": id_col_name_val,
                    "value": "num",
                    "col_name_vec": col_name_vec,
                    "value_vec": col_name_vec
                }

                # 将文档数据插入Elasticsearch索引
                es.index(index=index_name, doc_type='_doc', body=doc)
                
                
    print("数据已成功插入Elasticsearch索引：", index_name)

if __name__=="__main__":
    
    # 示例DataFrame
    df = pd.DataFrame({
        '学校': ['清华大学', '北京大学', '对外经济贸易大学'],
        '地址': ['北京', '湖南', '河北'],
        '学生数': [128, 211, 985]
    })
    # 拆分并构造Elasticsearch索引
    index_name = "try_sql_index"  # 设置Elasticsearch索引的名称
    id_col_name = "学校"  # 用户提供的id列名
    
    insert_into_es(df,es,index_name,id_col_name)