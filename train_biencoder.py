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
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, losses, InputExample # type: ignore
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
from typing import Dict, Iterable, List, Optional, TypedDict, Union
from pathlib import Path

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
    queries = get_queries(data_folder)
    ce_scores = get_ce_scores(data_folder)
    hard_negatives = load_hard_negatives(data_folder)
    train_queries = get_train_queries(hard_negatives, ce_scores, ce_score_margin, queries, negs_to_use, num_negs_per_system=num_negs_per_system, use_all_queries=use_all_queries)
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

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


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
parser.add_argument("--max_passages", default=0, type=int)
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


# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = 'output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


corpus = load_corpus(data_folder=data_folder)
train_queries = load_train_queries(data_folder=data_folder)
logging.info("Train queries: {}".format(len(train_queries)))


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params = {'lr': args.lr},
          )

# Save the model
model.save(model_save_path)