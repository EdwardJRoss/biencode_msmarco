# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py
"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.
For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)
Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.
With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus
Running this script:
python train_bi-encoder-v3.py
"""
import argparse
import gzip
import json
import logging
import os
import pickle
import random
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypedDict, Union

import torch
import torch.nn.functional as F
import tqdm
from sentence_transformers import LoggingHandler, util  # type: ignore
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoTokenizer,  # type: ignore
                          get_linear_schedule_with_warmup)

QueryId = int
PassageId = int
HardNegative = TypedDict('HardNegative', {'qid': QueryId, 'pos': List[PassageId], 'neg': Dict[str, List[PassageId]]})
TrainQuery = TypedDict('TrainQuery', {'qid': QueryId, 'query': str, 'pos': List[PassageId], 'neg': List[PassageId]})


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'

MSMARCO_COLLECTION_URL = "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz"
MSMARCO_QUERY_URL = "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz"

def download_and_unpack(url, output_path):
    if not os.path.exists(output_path):
        logging.info("Download {} and unpack to {}".format(url, output_path))
        util.http_get(url, output_path)
    with tarfile.open(output_path, "r:gz") as tar:
        tar.extractall(path=data_folder)

def read_int_value_tsv(filepath: Union[Path, str]) -> Dict[int, str]:
    """
    Read a tsv file and return a dict with the first column as key and the second column as value
    """
    data = {}
    with open(filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            key, value = line.strip().split("\t")
            data[int(key)] = value
    return data

def get_train_queries(hard_negatives: Iterable[HardNegative],
                      ce_scores: Dict[QueryId, Dict[PassageId, float]],
                      ce_score_margin: float,
                      queries: Dict[QueryId, str],
                      negs_to_use: Optional[List[str]] = None,
                      num_negs_per_system: int = 5,
                      use_all_queries: bool = False) -> Dict[QueryId, TrainQuery]:
    train_queries: Dict[QueryId, TrainQuery] = {}
    for data in hard_negatives:
        qid = data['qid']
        pos_pids = data['pos']

        if len(pos_pids) == 0:  #Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pos_pids, 'neg': list(neg_pids)}
    return train_queries

def load_corpus(data_folder=data_folder,
               collection_url=MSMARCO_COLLECTION_URL) -> Dict[PassageId, str]:
    """Read the corpus files, that contain all the passages.
    
    Store them in the corpus dict"""
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        download_and_unpack(collection_url, tar_filepath)
    logging.info("Read corpus: collection.tsv")
    return read_int_value_tsv(collection_filepath)


def get_queries(data_folder=data_folder,
                queries_url=MSMARCO_QUERY_URL) -> Dict[QueryId, str]:
    """Read the train queries, store in queries dict"""
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        download_and_unpack(queries_url, tar_filepath)
    return read_int_value_tsv(queries_filepath)

def get_ce_scores(data_folder=data_folder) -> Dict[QueryId, Dict[PassageId, float]]:
    """Load cross encoder scores.
    
    Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    """
    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)
    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)
    return ce_scores

def load_hard_negatives(data_folder=data_folder) -> Iterable[HardNegative]:
    """Load hard negatives file.
    
    As training data we use hard-negatives that have been mined using various systems"""
    hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
    if not os.path.exists(hard_negatives_filepath):
        logging.info("Download cross-encoder scores file")
        util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)

    logging.info("Read hard negatives train file")
    with gzip.open(hard_negatives_filepath, 'rt') as fIn:
        for line in tqdm.tqdm(fIn):
            yield json.loads(line)

def load_train_queries(data_folder=data_folder,
                       num_negs_per_system=5,
                       ce_score_margin=0.1,
                       negs_to_use=None,
                       use_all_queries=False) -> Dict[QueryId, TrainQuery]:
    cache_path = os.path.join(data_folder, f'train_queries_{num_negs_per_system}_{ce_score_margin}_{negs_to_use}_{use_all_queries}.pkl.gz')
    if not os.path.exists(cache_path):
        queries = get_queries(data_folder)
        ce_scores = get_ce_scores(data_folder)
        hard_negatives = load_hard_negatives(data_folder)
        train_queries = get_train_queries(hard_negatives, ce_scores, ce_score_margin, queries, negs_to_use, num_negs_per_system=num_negs_per_system, use_all_queries=use_all_queries)
        with gzip.open(cache_path, 'wb') as fOut:
            pickle.dump(train_queries, fOut)
    logging.info("Load train queries")
    with gzip.open(cache_path, 'rb') as fIn:
        train_queries = pickle.load(fIn)
    return train_queries

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return [query_text, pos_text, neg_text]

    def __len__(self):
        return len(self.queries)
    
class SentenceCollate:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        batch_parts = list(zip(*batch))
        batch_tokens = [self.tokenizer(text, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors='pt',
                                ) for text in batch_parts]
        return batch_tokens

def embed(model, tokens, normalize=True):
    hidden_state = model(**tokens).last_hidden_state
    embeddings = (hidden_state * tokens['attention_mask'].unsqueeze(-1)).sum(1) / tokens['attention_mask'].sum(1, keepdim=True)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

if __name__ == '__main__':
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=300, type=int)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_queries", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--ce_score_margin", default=3.0, type=float)
    args = parser.parse_args()

    print(args)

    # The  model we want to fine-tune
    model_name = args.model_name


    max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
    ce_score_margin = args.ce_score_margin             #Margin for the CrossEncoder score between negative and positive passages
    num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
    num_epochs = args.epochs                 # Number of epochs we want to train
    negs_to_use = args.negs_to_use.split(",") if args.negs_to_use is not None else None

    temperature = 0.05


    model = AutoModel.from_pretrained(model_name).cuda().train()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_save_path = 'output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


    corpus = load_corpus(data_folder=data_folder)
    train_queries = load_train_queries(data_folder=data_folder)
    if args.max_queries > 0:
        train_queries = {qid: train_queries[qid] for qid in list(train_queries.keys())[:args.max_queries]}
    logging.info("Train queries: {}".format(len(train_queries)))


    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataset = MSMARCODataset(train_queries, corpus=corpus)
    collator = SentenceCollate(tokenizer, max_length=max_seq_length)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collator)


    scaler = torch.cuda.amp.GradScaler()

    optimiser = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataloader) * num_epochs)

    for epoch in tqdm.trange(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm.tqdm(train_dataloader)):
            optimiser.zero_grad()

            query_tokens, pos_tokens, neg_tokens = batch

            query_tokens = {k:v.to(model.device) for k,v in query_tokens.items()}
            pos_tokens = {k:v.to(model.device) for k,v in pos_tokens.items()}
            neg_tokens = {k:v.to(model.device) for k,v in neg_tokens.items()}


            with torch.cuda.amp.autocast(dtype=torch.float16):
                query_emb = embed(model, query_tokens)
                pos_emb = embed(model, pos_tokens)
                neg_emb = embed(model, neg_tokens)
                doc_emb = torch.cat([pos_emb, neg_emb], dim=0)

                scores = (query_emb @ doc_emb.t()) / temperature
                labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)
            
                loss = F.cross_entropy(scores, labels)

            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            # Only step when optimiser isn't skipped
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/10   
            if not (scale > scaler.get_scale()):
                scheduler.step()
            if step % 100 == 0:
                logging.info("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, loss.item()))


    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)