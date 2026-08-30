"""" Datensatz: tpot_dataset/network_traffic.csv """

import os
import sys

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

# folder + file for the T-Pot dataset
DATA_DIR = os.environ.get("TPOT_DATA_DIR", "tpot_dataset")
TRAFFIC_FILE = os.path.join(DATA_DIR, "network_traffic.csv")

rng = np.random.default_rng(seed=42)

LOCAL_PREFIX = "10.42.0."

FEATURES = [
    "count",
    "dst_is_ip",
    "num_tags",
    "is_local_src",
    "is_local_dst",
]
INPUT_DIM = 5

# Helper functions
def ip_octets(value) -> tuple[int, int, int, int, int]:

    parts = str(value).strip().split(".")
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        o1, o2, o3, o4 = (int(p) for p in parts)
        return o1, o2, o3, o4, 1

    return 0, 0, 0, 0, 0

def extract_host(value) -> str:
    s = str(value).strip()
    if "://" in s:
        host = urlparse(s).hostname
        return host if host else s
    if ":" in s:
        head, _, tail = s.rpartition(":")
        if tail.isdigit() and head:
            return head
    return s

def is_local(value) -> int:
    # 1 if address is inside local subnet
    return 1 if str(value).startswith(LOCAL_PREFIX) else 0

# Feature creation
def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    raw_count = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    dst_host = df["url"].apply(extract_host)
    dst = dst_host.apply(ip_octets)
    df["dst_is_ip"] = dst.apply(lambda t: t[4])

    df["num_tags"] = df["tags"].fillna("").apply( lambda t: len(str(t).split(",")) if str(t).strip() else 0)
    df["is_local_src"] = df["src_ip"].apply(is_local)
    df["is_local_dst"] = dst_host.apply(is_local)
    df["count"] = np.log1p(raw_count)

    # ground truth label
    df["target"] = (df["label"].astype(str).str.strip().str.upper() == "MALICIOUS").astype(int)

    return df[FEATURES + ["target"]]

def scale_features(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32).copy()
    idx = {name: i for i, name in enumerate(FEATURES)}

    X[:, idx["num_tags"]] = np.clip(X[:, idx["num_tags"]], 0, 10) / 10.0
    X[: , idx["count"]] = X[: , idx["count"]] / 20.0

    return X



# Loading the data and building splits
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    raw = pd.read_csv(TRAFFIC_FILE)
    featurized = create_features(raw)
    benign = featurized[featurized["target"] == 0].drop(columns=["target"]).reset_index(drop=True)
    malicious = featurized[featurized["target"] == 1].drop(columns=["target"]).reset_index(drop=True)

    return benign, malicious

def partition(df: pd.DataFrame, partition_id: int, num_partitions: int) -> pd.DataFrame:
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    shard_indices = np.array_split(np.arange(len(shuffled)), num_partitions)
    return shuffled.iloc[shard_indices[partition_id]].reset_index(drop=True)

def build_splits(benign: pd.DataFrame, malicious: pd.DataFrame):

    benign_train, benign_test = train_test_split( benign, test_size=0.2, random_state=42)
    benign_train_ae, benign_val_ae = train_test_split( benign_train, test_size=0.2, random_state=42)
    benign_train_ae, benign_for_classifier = train_test_split( benign_train_ae, test_size=0.2, random_state=42)

    if len(malicious) >= 4:
        mal_train, mal_test = train_test_split(malicious, test_size=0.2, random_state=42)
    else:
        mal_train, mal_test = malicious, malicious # too few malicious samples

    X_train = scale_features(benign_train_ae[FEATURES].to_numpy())

    X_Validation = scale_features(benign_val_ae[FEATURES].to_numpy())

    X_train_classifier = scale_features(pd.concat([benign_for_classifier, mal_train], ignore_index=True)[FEATURES].to_numpy())

    y_class = np.concatenate( [np.zeros(len(benign_for_classifier)), np.ones(len(mal_train))])

    X_test_full = scale_features(pd.concat([benign_test, mal_test], ignore_index=True)[FEATURES].to_numpy())

    y_true = np.concatenate([np.zeros(len(benign_test)), np.ones(len(mal_test))])

    trainloader = DataLoader( TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)

    validaton_loader = DataLoader( TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64, shuffle=True)

    return trainloader, validaton_loader, X_test_full, X_Validation, y_true, X_train_classifier, y_class




def load_mono_dataset(partition_id: int, num_partitions: int):

    benign, malicious = load_data()
    benign_part = partition(benign, partition_id, num_partitions)
    malicious_part = partition(malicious, partition_id, num_partitions)
    return build_splits(benign_part, malicious_part)

def load_centralized_dataset():
    benign, malicious = load_data()
    trainloader, validaton_loader, X_test_full,X_Validation, y_true, X_train_classifier, y_class = (build_splits(benign, malicious))

    return trainloader, validaton_loader, X_test_full, X_Validation, y_true, X_train_classifier, y_class

#Todo

# def load_cross_data()