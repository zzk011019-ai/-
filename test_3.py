# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import math
import json
import os
from deep_translator import GoogleTranslator
import time

# --------------------------
# 1. 加载语料
# --------------------------
path = "preprocessed_data/ch_ko.csv"
df = pd.read_csv(path)

ch_queries = df['zh-cn'].astype(str).tolist()
kr_docs = df['ko'].astype(str).tolist()
num_docs = len(kr_docs)
print("词表",num_docs)

# --------------------------
# 2. 查询翻译函数
# --------------------------
def translate_query_to_korean(query: str) -> str:
    """
    使用翻译 API 将中文 query 翻译为韩语。
    返回翻译后的韩语文本。
    """
    try:
        # “zh‑CN” → “ko” 表示 中文（简体）→ 韩语
        translated = GoogleTranslator(source='zh-CN', target='ko').translate(query)
        return translated
    except Exception as e:
        print(f"[翻译失败] 原文: {query}，错误: {e}")
        # 回退机制：返回原文或保留未翻译状态
        return query

# 英语中介翻译
def translate_ch_to_english(query: str) -> str:
    """
    使用 GoogleTranslator 将中文翻译成英文
    """
    try:
        translated = GoogleTranslator(source='zh-CN', target='en').translate(query)
        return translated
    except Exception as e:
        print(f"[翻译失败] 原文: {query}, 错误: {e}")
        return query  # 回退到原文
    
def translate_en_to_korean(query_en: str) -> str:
    """
    使用 GoogleTranslator 将英文句子翻译成韩文
    """
    try:
        translated = GoogleTranslator(source='en', target='ko').translate(query_en)
        return translated
    except Exception as e:
        print(f"[翻译失败] 原文: {query_en}, 错误: {e}")
        return query_en  # 回退到原文


# --------------------------
# 3. 加载模型 & 向量化
# --------------------------
print("加载模型")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("编码词表")
# 只取前 200 条文档快速测试
kr_docs_sample = kr_docs[:200]

doc_embeddings = model.encode(
    kr_docs_sample,
    batch_size=64,         # 每批次 64 条文档，可根据显存调大
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True # 显示进度
)


# FAISS 索引
d = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(doc_embeddings)

# --------------------------
# 4. 检索函数
# --------------------------
def retrieve_vector(query_kr, top_k=5):
    query_emb = model.encode([query_kr], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_emb, top_k)
    return idxs[0], scores[0]

# --------------------------
# 5. 评价函数
# --------------------------
def calc_map(results, queries, ground_truth, k=5):
    ap_sum = 0.0
    for i, q in enumerate(queries):
        relevant = set(ground_truth[q])
        retrieved = results[i][:k]
        num_hits = 0
        score = 0.0
        for rank, doc_idx in enumerate(retrieved):
            if doc_idx in relevant:
                num_hits += 1
                score += num_hits / (rank+1)
        if len(relevant) > 0:
            ap_sum += score / len(relevant)
    return ap_sum / len(queries)

def calc_p_at_k(results, queries, ground_truth, k=5):
    p_sum = 0.0
    for i, q in enumerate(queries):
        relevant = set(ground_truth[q])
        retrieved = results[i][:k]
        num_rel_ret = sum([1 for doc in retrieved if doc in relevant])
        p_sum += num_rel_ret / k
    return p_sum / len(queries)

def calc_ndcg_at_k(results, queries, ground_truth, k=5):
    ndcg_sum = 0.0
    for i, q in enumerate(queries):
        relevant = set(ground_truth[q])
        retrieved = results[i][:k]
        dcg = 0.0
        for rank, doc_idx in enumerate(retrieved):
            if doc_idx in relevant:
                dcg += 1 / math.log2(rank + 2)
        idcg = sum([1 / math.log2(r + 2) for r in range(min(len(relevant), k))])
        if idcg > 0:
            ndcg_sum += dcg / idcg
    return ndcg_sum / len(queries)

# --------------------------
# 6. ground_truth
# --------------------------
ch_queries = ch_queries[:20]
ground_truth = {ch_queries[i]: [i] for i in range(len(ch_queries))}

# --------------------------
# 7. 执行检索实验
# --------------------------
methods = ["dict+extension+transliteration", "simple_dict", "english_intermediate"]
all_results = {}
top_k = 5
for method in methods:
    print(f"\n=== 开始方法: {method} ===")
    method_results = []
    start_time = time.time()
    for q in ch_queries:
        if method == "english_intermediate":
            q_en = translate_ch_to_english(q)
            q_kr = translate_en_to_korean(q_en)
        else:
            q_kr = translate_query_to_korean(q)
        idxs, scores = retrieve_vector(q_kr, top_k=top_k)
        method_results.append({
            "query": q,
            "query_kr": q_kr,
            "retrieved_idxs": idxs.tolist(),
            "scores": scores.tolist()
        })
    all_results[method] = method_results
    print(f"{method} 方法完成，用时 {time.time()-start_time:.2f}s")


# --------------------------
# 8. 计算评价指标并保存
# --------------------------
metrics = {}
for method, res in all_results.items():
    retrieved_idxs = [r['retrieved_idxs'] for r in res]
    map_score = calc_map(retrieved_idxs, ch_queries, ground_truth, k=top_k)
    p_score = calc_p_at_k(retrieved_idxs, ch_queries, ground_truth, k=top_k)
    ndcg_score = calc_ndcg_at_k(retrieved_idxs, ch_queries, ground_truth, k=top_k)
    metrics[method] = {"MAP": map_score, f"P@{top_k}": p_score, f"nDCG@{top_k}": ndcg_score}

# 保存结果
os.makedirs("results", exist_ok=True)
with open("results/retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

with open("results/retrieval_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("实验完成，结果已保存到 results/ 目录")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
