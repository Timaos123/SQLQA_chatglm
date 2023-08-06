from elasticsearch import Elasticsearch
# from sentence_transformers import SentenceTransformer
from transformers import AutoModel,AutoTokenizer
import http.client
import json
import requests
import torch
import pandas as pd
from new_A0_try_input_Text2SQL import insert_into_es
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode_text(bert_model, bert_tokenizer, text):
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

def search_similar_vectors(query_text, indexName="sqlqa_index", 
                                                        simVal="content",
                                                        simVec="content_vector",
                                                        simType="column",top_k=5):
    # 返回近似字段
    # simType: column/value
    
    bert_model = AutoModel.from_pretrained("./bert-base-chinese")
    bert_tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese")
    query_vector=encode_text(bert_model, bert_tokenizer, query_text)
    del bert_model
    del bert_tokenizer
    
    if simType=="column":
        body = {"size": 0,
                        "aggs": {
                            "deduplicated_values": {
                                "terms": {
                                    "field": simVal,
                                    "size": top_k
                                },
                                "aggs": {
                                    "top_hits": {
                                    "top_hits": {
                                        "_source": {
                                        "includes": ["col_name"] 
                                        },
                                        "size": top_k
                                    }
                                    }
                                }
                            }
                        },
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, '{}') + 1.0".format(simVec),
                                    "params": {"query_vector": query_vector.flatten().tolist()}
                                }
                            }
                        }
                    }
        # # print(body)
        # response = es.search(index=indexName, body=body)["aggs"]["hits"]["hits"]
        # response=[row["_source"] for row in response]
    else:
        body = {
            "size": 0,
            "aggs": {
                "deduplicated_values": {
                    "terms": {
                        "field": simVal,
                        "size": top_k
                    },
                    "aggs": {
                        "top_hits": {
                        "top_hits": {
                            "_source": {
                            "includes": ["col_name", "value"] 
                            },
                            "size": top_k
                        }
                        }
                    }
                }
            },
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, '{}') + 1.0".format(simVec),
                        "params": {"query_vector": query_vector.flatten().tolist()}
                    }
                }
            }
        }
    buckets = es.search(index=indexName, body=body)["aggregations"]["deduplicated_values"]["buckets"]
    response=[]
    for bucketItem in buckets:
        response+=bucketItem["top_hits"]["hits"]["hits"]
    response=[row["_source"] for row in response]
    return response


def search_similar_text(query_text, 
                                            indexName="sqlqa_index", 
                                            simVal="col_name",
                                            simType="column", top_k=5):
    # simType: column/value
    if simType=="column":
        body = {
            "size": 0,
            "aggs": {
                "deduplicated_values": {
                    "terms": {
                        "field": simVal,
                        "size": top_k
                    },
                    "aggs": {
                        "top_hits": {
                        "top_hits": {
                            "_source": {
                            "includes": ["col_name"] 
                            },
                            "size": top_k
                        }
                        }
                    }
                }
            },
            "query": {
                "match": {
                    simVal: {
                        "query": query_text,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
        # # print(body)
        # response = es.search(index=indexName, body=body)["aggs"]["hits"]["hits"]
        # response=[row["_source"] for row in response]
    else:
        body = {
            "size": 0,
            "aggs": {
                "deduplicated_values": {
                    "terms": {
                        "field": simVal,
                        "size": top_k
                    },
                    "aggs": {
                        "top_hits": {
                        "top_hits": {
                            "_source": {
                            "includes": ["col_name", "value"] 
                            },
                            "size": top_k
                        }
                        }
                    }
                }
            },
            "query": {
                "match": {
                    simVal: {
                        "query": query_text,
                        "fuzziness": "2",
                        "prefix_length": 1,
                        "max_expansions": 50
                    }
                }
            }
        }
    buckets = es.search(index=indexName, body=body)["aggregations"]["deduplicated_values"]["buckets"]
    response=[]
    for bucketItem in buckets:
        response+=bucketItem["top_hits"]["hits"]["hits"]
    response=[row["_source"] for row in response]
    return response

def get_related_columns(user_text,indexName="sqlqa_index"):
    # 基于文本的向量近似度和关键词近似度处理逻辑
    
    # 近似列
    colList1=search_similar_vectors(user_text, 
                                                                    indexName=indexName,
                                                                    simType="column",
                                                                    simVec="col_name_vec",
                                                                    simVal="col_name", top_k=5)
    colList2=search_similar_text(user_text, 
                                                            indexName=indexName, 
                                                            simType="column",
                                                            simVal="col_name",
                                                            top_k=5)
    colList=colList1+colList2
    colList=[colItem["col_name"] for colItem in colList]
    colList=list(set(colList))
    
    return colList

def retrieve_sample_table(myDf,relevant_fields,sampleN=5):
    # 从数据库中获取样本表格的逻辑
    return myDf.loc[:,relevant_fields].sample(min(sampleN,myDf.shape[0]))

def chatGLM(prompt):
    import requests
    import json

    # url = "http://172.2.0.97:6006/beauty_industry_doc_qa"
    url = "http://172.2.0.97:6006/chatglm/generate_content"

    payload = json.dumps({
        "prompt":prompt
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': '172.2.0.97:6006',
        'Connection': 'keep-alive'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    
    return response.json()["data"]

def construct_base_sql(user_text,sample_table,table_name="school_df"):
    # 基于样本表格构建基础的SQL的逻辑
    prompt="我们拥有如下数据：\n {}:\n{}\n".format(table_name,sample_table.to_markdown()) + \
                    "请根据以上数据以及用户问题：'{}'\n".format(user_text)+ \
                    "构建SQL解答用户问题。\n"+\
                    "构建SQL的时候注意使用原表的字段，答案只需要sql，不需要别的解释，请用以下格式回答：\n"+\
                    "生成的SQL为：```你生成的SQL```\n"+\
                    "生成的SQL为："
    SQLResult=chatGLM(prompt)
    return SQLResult

def check_quotes(base_sql):
    if "'" in base_sql:
        return True
    else:
        return False

def get_related_values(user_text):
    # 在ES中查询与单引号内最相近的5个值的逻辑
    # 近似值
    valList1=search_similar_vectors(user_text, 
                                                                    indexName=indexName,
                                                                    simType="value",
                                                                    simVec="value_vec",
                                                                    simVal="value", top_k=3)
    valList2=search_similar_text(user_text, 
                                                            indexName=indexName,
                                                            simVal="value",
                                                            simType="value",
                                                            top_k=3)
    valList=valList1+valList2
    valList=[(row["col_name"],row["value"]) for row in valList]
    valList=list(set(map(lambda row:str(row),valList)))
    valList=list(map(lambda row:eval(row),valList))
    valList=list(filter(lambda row:row[1]!="num",valList))
    valDict=dict(valList)
    
    return valDict

def construct_new_table(sample_table, attr_val_dict):
    # 构建新的样例表格的逻辑
    matchEvalStr="|".join(["(sample_table['{}']=='{}')".format(k,v) for k,v in attr_val_dict.items()])
    new_table=sample_table.loc[eval(matchEvalStr),:]
    return new_table

def reconstruct_sql(base_sql, user_text):
    # 重构SQL，结合用户输入的文本的逻辑
    # ...
    return reconstructed_sql

import duckdb
def execute_sql(reconstructed_sql,myDf,table_name="school_df"):
    # 执行SQL检索的逻辑
    # df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})

    con = duckdb.connect()
    con.register(table_name, myDf)

    result = con.execute(reconstructed_sql.replace("\\n",""))
    df_result = result.fetchdf()
    
    return df_result

def generate_answer(user_text, newSQL,sample_table,query_result,table_name="school_df"):
    print(newSQL)
    print(query_result.to_markdown())
    # 结合用户输入的问题和SQL检索结果生成回答的逻辑
    prompt="所得结果为：\n{}".format(query_result.to_markdown())+\
                    "根据以上数据回答用户问题：'{}'\n".format(user_text)+ \
                    "你的回答是："
    answer=chatGLM(prompt)
    return answer

from fuzzywuzzy import process
def find_most_similar_string(str1, string_list):
    return process.extractOne(str1, string_list)[0]
    

if __name__=="__main__":
    # 用户输入文本
    user_text = input("请输入文本: ")
    # 示例DataFrame
    myDf = pd.DataFrame({
        'city': ['北京', '上海', '杭州'],
        'hotel_num': [4700, 3101, 1985]
    })
    indexName="hotel_index"
    id_col_name = "city"
    table_name="hotel_count"
    
    # 初始化Elasticsearch
    es = Elasticsearch(hosts=["http://47.98.173.193:9200"])
    
    insert_into_es(myDf,es,indexName,id_col_name)

    # 处理输入文本，获取相关字段
    relevant_fields = get_related_columns(user_text,indexName=indexName)

    # 从数据库中获取样本表格
    sample_table = retrieve_sample_table(myDf,relevant_fields)

    # 构建基础的SQL查询语句
    base_sql = construct_base_sql(user_text,sample_table,table_name=table_name)
    base_sql=base_sql.replace("\"","").replace("`","")

    # 判断SQL中是否存在单引号
    if check_quotes(base_sql):
        # 在ES中查询与单引号内最相近的5个值
        attr_val_dict = get_related_values(user_text)

        # 构建新的样例表格
        new_table = construct_new_table(sample_table, attr_val_dict)

        # 重构SQL查询语句，结合用户输入的文本
        sqlKVList=re.findall("\S*\s+=\s+'.*?'",base_sql)
        for sqlKVItem in sqlKVList:
            k,v=sqlKVItem.split("=")
            k=k.strip()
            if "." in k:
                k=k.split(".")[1]
            v=v.replace("'","").strip()
            newVList=new_table[k].values.flatten().tolist()
            newV=find_most_similar_string(v,newVList)
            base_sql=base_sql.replace(v,newV)
            
        reconstructed_sql = base_sql
    else:
        new_table = sample_table
        reconstructed_sql = base_sql

    # 执行SQL查询
    query_result = execute_sql(reconstructed_sql,sample_table,table_name=table_name)

    # 生成回答
    answer = generate_answer(user_text,reconstructed_sql, sample_table,query_result,table_name=table_name)

    # 输出结果
    if "你的回答是：" in answer:
        answer=answer.split("你的回答是：")[1]
        answer=answer.replace("\"","")
        
    print("回答:", answer)
        
    print("查询结果:", query_result)
