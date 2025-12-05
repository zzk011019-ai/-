"""
Full Pipeline Script for Simplified vs Traditional Chinese Experiments
- Automatically reads two JSON files (simplified & traditional)
- Generates sample_corpus, queries, and qrels from QA JSON data
- Runs Experiment 1 (network analysis) and Experiment 2 (retrieval)
- Requires: PyTorch, transformers, networkx, jieba, opencc-python-reimplemented
"""

from simplified_vs_traditional_experiments import experiment_network, experiment_retrieval
import os
import json

# --------- JSON file paths ---------
simplified_json_path = "cmrc2018_train.json"
traditional_json_path = "DRCD_training.json"

# --------- Parse JSON to corpus ---------
sample_corpus = {"simplified": [], "traditional": []}


def parse_json_to_corpus(json_path, key):
    corpus_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):  
        entries = data.get("data", [])
        for entry in entries:
            title = entry.get("title", "unknown")
            for para in entry.get("paragraphs", []):
                context = para.get("context", "").strip()
                if context:
                    docid = para.get("id", f"{title}-para")
                    corpus_list.append((docid, context))
    elif isinstance(data, list):  
        for entry in data:
            title = entry.get("title", "unknown")
            context_id = entry.get("context_id", f"{title}-para")
            context_text = entry.get("context_text", "").strip()
            if context_text:
                corpus_list.append((context_id, context_text))
    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(data)}")

    sample_corpus[key] = corpus_list


parse_json_to_corpus(simplified_json_path, "simplified")
parse_json_to_corpus(traditional_json_path, "traditional")

# --------- Generate queries and qrels from QA ---------
queries = []
qrels = {}

def generate_queries_and_qrels(json_path, region_key):
    """
    生成 queries 和 qrels，兼容两种 JSON 顶层格式
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    # with open("data/zh_simplified.json", 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    print(type(data))  # dict 或 list
    if isinstance(data, dict):
        print(data.keys())  # dict 的 key
    elif isinstance(data, list):
        print("Top-level is list, first element:")
        print(data[0].keys())


    if isinstance(data, dict):
        entries = data.get("data", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(data)}")

    for entry in entries:
        for para in entry.get("paragraphs", []):
            docid = para.get("id")
            for qa in para.get("qas", []):
                qid = qa.get("id")
                question = qa.get("question")
                queries.append((qid, question))
                qrels[qid] = {f"{region_key}::{docid}"}


generate_queries_and_qrels(simplified_json_path, "simplified")
generate_queries_and_qrels(traditional_json_path, "traditional")

# --------- Create temporary folder structure ---------
tmp_dir = 'tmp_sample_data'
os.makedirs(tmp_dir, exist_ok=True)

# print("Simplified corpus sample:")
# for i, (docid, text) in enumerate(sample_corpus["simplified"][:5]):
#     print(docid, repr(text))


for key, docs in sample_corpus.items():
    region_path = os.path.join(tmp_dir, key)
    os.makedirs(region_path, exist_ok=True)
    with open(os.path.join(region_path, 'corpus.txt'), 'w', encoding='utf-8') as f:
        for docid, text in docs:
            f.write(f"{docid}\t{text}\n")

queries_path = os.path.join(tmp_dir, 'queries.txt')
with open(queries_path, 'w', encoding='utf-8') as f:
    for qid, text in queries:
        f.write(f"{qid}\t{text}\n")

qrels_path = os.path.join(tmp_dir, 'qrels.txt')
with open(qrels_path, 'w', encoding='utf-8') as f:
    for qid, docids in qrels.items():
        for docid in docids:
            f.write(f"{qid}\t{docid}\t1\n")

# --------- Run Experiment 1: Network Analysis ---------
print("===== Experiment 1: Network Analysis (Simplified corpus, word-level) =====")
corpus_path = os.path.join(tmp_dir, 'simplified', 'corpus.txt')
G_s, metrics_s = experiment_network(corpus_path, convert='none', level='word', window=2)
print(metrics_s)

print("\n===== Experiment 1: Network Analysis (Traditional corpus, word-level) =====")
corpus_path = os.path.join(tmp_dir, 'traditional', 'corpus.txt')
G_t, metrics_t = experiment_network(corpus_path, convert='none', level='word', window=2)
print(metrics_t)
out_file = os.path.join(tmp_dir, "network_metrics.txt")

with open(out_file, "w", encoding="utf-8") as f:
    f.write("===== Experiment 1: Network Analysis (Simplified corpus, word-level) =====\n")
    f.write(json.dumps(metrics_s, ensure_ascii=False, indent=2))
    f.write("\n\n")  # 分隔两个实验结果
    f.write("===== Experiment 1: Network Analysis (Traditional corpus, word-level) =====\n")
    f.write(json.dumps(metrics_t, ensure_ascii=False, indent=2))

print(f"Network metrics saved to {out_file}")

# --------- Run Experiment 2: Retrieval ---------
print("\n===== Experiment 2: Retrieval (Simplified & Traditional, traditional->simplified) =====")
pred_ids, recall = experiment_retrieval(
    corpus_dir=tmp_dir,
    queries_path=queries_path,
    qrels_path=qrels_path,
    convert='t2s',  # traditional->simplified
    encoder_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    k=5,
    sample=None
)
print("Recall@5:", recall)
print("Predicted top docs per query:", json.dumps(pred_ids, ensure_ascii=False, indent=2))

print("\nTemporary data stored in:", tmp_dir)
