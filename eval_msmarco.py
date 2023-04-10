# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/eval_msmarco.py
"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.
Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""

import torch
import torch.nn.functional as F
from sentence_transformers import  LoggingHandler, util
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
import sys
import os
from tqdm import tqdm
import tarfile

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Name of the SBERT model
model_name = sys.argv[1]

# You can limit the approx. max size of the corpus. Pass 100 as second parameter and the corpus has a size of approx 100k docs
corpus_max_size = int(sys.argv[2])*1000 if len(sys.argv) >= 3 else 0


####  Load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(model_name).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

### Data files
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

collection_filepath = os.path.join(data_folder, 'collection.tsv')
dev_queries_file = os.path.join(data_folder, 'queries.dev.small.tsv')
qrels_filepath = os.path.join(data_folder, 'qrels.dev.tsv')

### Download files if needed
if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
    tar_filepath = os.path.join(data_folder, 'collectionandqueries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download: "+tar_filepath)
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


if not os.path.exists(qrels_filepath):
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv', qrels_filepath)

### Load data

corpus = {}             #Our corpus pid => passage
dev_queries = {}        #Our dev queries. qid => query
dev_rel_docs = {}       #Mapping qid => set with relevant pids
needed_pids = set()     #Passage IDs we need
needed_qids = set()     #Query IDs we need

# Load the 6980 dev queries
with open(dev_queries_file, encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        dev_queries[qid] = query.strip()


# Load which passages are relevant for which queries
with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split('\t')

        if qid not in dev_queries:
            continue

        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)


# Read passages
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        passage = passage

        if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
            corpus[pid] = passage.strip()



## Run evaluator
logging.info("Queries: {}".format(len(dev_queries)))
logging.info("Corpus: {}".format(len(corpus)))

k = 10
max_seq_length = 512

def embed_passages(passages, batch_size=100, normalize=True):
    embeddings = []
    for i in tqdm(range(0, len(passages), batch_size), desc="Embedding passages", total=len(passages)//batch_size):
        batch = passages[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        with torch.inference_mode():
            hidden_state = model(**tokens).last_hidden_state
            # Mean pooling, only on attended tokens
            batch_embeddings = (hidden_state * tokens['attention_mask'].unsqueeze(-1)).sum(dim=1) / tokens['attention_mask'].sum(dim=1, keepdim=True)
            if normalize:
                batch_embeddings = F.normalize(batch_embeddings, dim=1)
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.concatenate(embeddings)


logging.info("Embedding queries")
query_embeddings = embed_passages(list(dev_queries.values()), normalize=True)
logging.info("Embedding passages")
corpus_embeddings = embed_passages(list(corpus.values()), normalize=True)

scores = query_embeddings @ corpus_embeddings.T
top_idx = np.argsort(-scores, axis=1)[:, :k]
query_preds = {qid: [list(corpus.keys())[idx] for idx in top_idx[i]] for i, qid in enumerate(dev_queries.keys())}

def rr(y_true, y_pred):
    common = set(y_true).intersection(y_pred)
    if not common:
        return 0
    return 1 / (min([y_pred.index(pid) for pid in common]) + 1)

mrr_score = np.mean([rr(dev_rel_docs[qid], query_preds[qid]) for qid in query_preds])
logging.info("MRR@{}: {:.4f}".format(k, mrr_score))
recall_at_k = np.mean([len(set(dev_rel_docs[qid]).intersection(set(query_preds[qid]))) / len(dev_rel_docs[qid]) for qid in query_preds])
logging.info("Recall@{}: {:.4f}".format(k, recall_at_k))