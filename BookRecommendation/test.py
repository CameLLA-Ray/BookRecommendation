"""
Book Recommendation System - 3 Algorithms + Metrics (Precision, Recall, NDCG)

This script implements three recommenders:
 1) User-based KNN (lecture algorithm)
 2) LightGBM (pointwise prediction with negative sampling)
 3) BPR (Bayesian Personalized Ranking) using `implicit`

After training, each model outputs Precision@K, Recall@K, and NDCG@K.

Expected input files (downloaded from the Kaggle dataset and placed in the same folder):
 - "BX-Users.csv" or "users.csv"
 - "BX-Books.csv" or "books.csv"
 - "BX-Book-Ratings.csv" or "ratings.csv"

Install requirements:
 pip install pandas numpy scipy scikit-learn lightgbm implicit

Usage:
 python book_recommender_with_metrics.py

Note: change file paths at the top if your filenames differ.
"""

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# BPR from implicit
try:
    from implicit.bpr import BayesianPersonalizedRanking
except Exception as e:
    raise ImportError(
        "Package 'implicit' is required for BPR. Install with 'pip install implicit'.\n" + str(e)
    )

# ----------------------------
# Config / file paths
# ----------------------------
USERS_FILE = "Users.csv"  # or users.csv
BOOKS_FILE = "Books.csv"  # or books.csv
RATINGS_FILE = "Ratings.csv"  # or ratings.csv

# Evaluation top-K
K = 5

# ----------------------------
# Utilities: load dataset
# ----------------------------

def load_kaggle_book_dataset(users_file=USERS_FILE, books_file=BOOKS_FILE, ratings_file=RATINGS_FILE):
    """Robust loader: tries common CSV names and separators."""
    def _try_read(f):
        if not os.path.exists(f):
            return None
        # Try reading with common encodings and separators
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(f, sep=sep, encoding='latin-1')
                return df
            except Exception:
                continue
        # fallback
        return pd.read_csv(f, encoding='latin-1')

    users = _try_read(users_file)
    books = _try_read(books_file)
    ratings = _try_read(ratings_file)

    if users is None or books is None or ratings is None:
        raise FileNotFoundError(
            f"One or more input files not found. Check paths: {users_file}, {books_file}, {ratings_file}"
        )

    return users, books, ratings


# ----------------------------
# Preprocessing and split
# ----------------------------

def preprocess_and_split(users, books, ratings, test_size=0.2, seed=42):
    """
    - Merge metadata where useful
    - Build interaction DataFrame with columns: user_id, book_id, rating
    - Split per-user: leave-one-out or ratio split so each user has test interactions
    """
    # Standardize column names if they are prefixed with 'Book-' etc.
    ratings = ratings.rename(columns=lambda c: c.strip().replace(' ', '_'))
    users = users.rename(columns=lambda c: c.strip().replace(' ', '_'))
    books = books.rename(columns=lambda c: c.strip().replace(' ', '_'))

    # Detect key column names
    user_col = None
    item_col = None
    rating_col = None
    for c in ratings.columns:
        lc = c.lower()
        if 'user' in lc and user_col is None:
            user_col = c
        if 'book' in lc and 'isbn' in lc.lower():
            item_col = c
        if 'isbn' in lc.lower() and item_col is None:
            item_col = c
        if 'rating' in lc and rating_col is None:
            rating_col = c

    if user_col is None or item_col is None or rating_col is None:
        # fallback guesses
        possible = list(ratings.columns)
        user_col, item_col, rating_col = possible[0], possible[1], possible[2]

    df = ratings[[user_col, item_col, rating_col]].copy()
    df.columns = ['user_id', 'book_id', 'rating']

    # Drop missing
    df = df.dropna(subset=['user_id', 'book_id'])

    # Convert to numeric ids
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'].astype(str))
    df['item_idx'] = item_encoder.fit_transform(df['book_id'].astype(str))

    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()
    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")

    # Per-user split: for each user, sample test interactions
    rng = np.random.RandomState(seed)
    train_list = []
    test_list = []
    grouped = df.groupby('user_idx')
    for u, g in grouped:
        items = g.index.values
        if len(items) == 1:
            train_list.append(items[0])
            continue
        # choose proportionally
        test_count = max(1, int(len(items) * test_size))
        test_idx = rng.choice(items, size=test_count, replace=False)
        for idx in items:
            if idx in test_idx:
                test_list.append(idx)
            else:
                train_list.append(idx)

    train_df = df.loc[train_list].reset_index(drop=True)
    test_df = df.loc[test_list].reset_index(drop=True)

    # Build sparse interaction matrix for training
    train_matrix = sparse.csr_matrix((train_df['rating'].astype(float), (train_df['user_idx'], train_df['item_idx'])))

    return {
        'train_df': train_df,
        'test_df': test_df,
        'train_matrix': train_matrix,
        'n_users': n_users,
        'n_items': n_items,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
    }


# ----------------------------
# Evaluation metrics: Precision@K, Recall@K, NDCG@K
# ----------------------------

def precision_at_k(recommended, ground_truth, k):
    if len(recommended) > k:
        recommended = recommended[:k]
    hits = len(set(recommended) & set(ground_truth))
    return hits / k


def recall_at_k(recommended, ground_truth, k):
    if len(ground_truth) == 0:
        return 0.0
    if len(recommended) > k:
        recommended = recommended[:k]
    hits = len(set(recommended) & set(ground_truth))
    return hits / len(ground_truth)


def dcg_at_k(recommended, ground_truth, k):
    recommended = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i starts at 0
    return dcg


def idcg_at_k(ground_truth, k):
    # ideal DCG when all ground truth items are at top
    ideal_hits = min(len(ground_truth), k)
    idcg = sum((1.0 / np.log2(i + 2)) for i in range(ideal_hits))
    return idcg


def ndcg_at_k(recommended, ground_truth, k):
    idcg = idcg_at_k(ground_truth, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, ground_truth, k) / idcg


def evaluate_model(recommender_fn, train_df, test_df, n_users, n_items, k=K):
    """
    recommender_fn: function(user_idx, k) -> list of item_idx recommended
    Returns average precision, recall, ndcg over users with test interactions
    """
    # Build ground truth per user from test_df
    gt = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    users = list(gt.keys())
    precisions = []
    recalls = []
    ndcgs = []
    for u in users:
        recs = recommender_fn(u, k)
        ground = gt.get(u, [])
        precisions.append(precision_at_k(recs, ground, k))
        recalls.append(recall_at_k(recs, ground, k))
        ndcgs.append(ndcg_at_k(recs, ground, k))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


# ----------------------------
# 1) User-based KNN recommender
# ----------------------------
class UserKNNRecommender:
    def __init__(self, train_matrix, n_neighbors=20):
        # train_matrix: sparse csr (users x items)
        self.train_matrix = train_matrix.tocsr()
        self.n_users, self.n_items = train_matrix.shape
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
        # Fit on user vectors
        self.model.fit(self.train_matrix)

    def recommend(self, user_idx, k=K):
        # find similar users
        user_vec = self.train_matrix[user_idx]
        # If user has no history, recommend popular items
        if user_vec.getnnz() == 0:
            return self._popular_items(k)
        dists, neighbors = self.model.kneighbors(user_vec, return_distance=True)
        neighbors = neighbors.flatten()
        # aggregate neighbor scores
        scores = defaultdict(float)
        for n in neighbors:
            row = self.train_matrix[n]
            nz = row.nonzero()[1]
            for i in nz:
                scores[i] += row[0, i]
        # remove items already seen by user
        seen = set(self.train_matrix[user_idx].nonzero()[1])
        ranked = [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True) if i not in seen]
        return ranked[:k]

    def _popular_items(self, k):
        sums = np.array(self.train_matrix.sum(axis=0)).ravel()
        top = np.argsort(-sums)[:k]
        return top.tolist()


# ----------------------------
# 2) LightGBM pointwise recommender with negative sampling
# ----------------------------
class LightGBMRecommender:
    def __init__(self, train_df, user_meta=None, item_meta=None, n_neg=3, seed=42):
        self.train_df = train_df
        self.user_meta = user_meta
        self.item_meta = item_meta
        self.n_neg = n_neg
        self.seed = seed
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def _make_features(self, df):
        # Basic features: user_idx, item_idx
        X = pd.DataFrame()
        X['user_idx'] = df['user_idx'].astype(int)
        X['item_idx'] = df['item_idx'].astype(int)
        # Optionally, include side info if available
        # Label-encode and return
        return X

    def fit(self, n_iter=100):
        # Create negative samples
        rng = np.random.RandomState(self.seed)
        users = self.train_df['user_idx'].unique()
        all_items = self.train_df['item_idx'].unique()
        neg_rows = []
        user_item_set = set(zip(self.train_df['user_idx'], self.train_df['item_idx']))
        for (u, g) in self.train_df.groupby('user_idx'):
            pos_items = set(g['item_idx'].values)
            for _ in range(self.n_neg * len(pos_items)):
                neg = rng.choice(all_items)
                if neg in pos_items:
                    continue
                neg_rows.append({'user_idx': u, 'item_idx': neg, 'rating': 0})

        train_neg = pd.DataFrame(neg_rows)
        train_all = pd.concat([self.train_df[['user_idx', 'item_idx', 'rating']], train_neg], ignore_index=True)
        X = self._make_features(train_all)
        y = (train_all['rating'] > 0).astype(int)

        # LightGBM dataset
        lgb_train = lgb.Dataset(X, label=y)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': self.seed
        }
        self.model = lgb.train(params, lgb_train, num_boost_round=100)

        # For faster recommendation, precompute item list
        self.all_items = np.unique(self.train_df['item_idx'].values)

    def recommend(self, user_idx, k=K):
        # Predict scores for all items, excluding seen ones
        seen = set(self.train_df[self.train_df['user_idx'] == user_idx]['item_idx'].values)
        candidates = [i for i in self.all_items if i not in seen]
        if len(candidates) == 0:
            return []
        Xcand = pd.DataFrame({'user_idx': [user_idx] * len(candidates), 'item_idx': candidates})
        preds = self.model.predict(Xcand)
        ranked = [item for _, item in sorted(zip(preds, candidates), key=lambda x: x[0], reverse=True)]
        return ranked[:k]


# ----------------------------
# 3) BPR recommender (implicit lib)
# ----------------------------
class BPRRecommender:
    def __init__(self, train_matrix, factors=64, iterations=50, learning_rate=0.05):
        # implicit expects item-user matrix
        self.train_matrix = train_matrix.tocsr()
        self.model = BayesianPersonalizedRanking(factors=factors, learning_rate=learning_rate)
        self.iterations = iterations

    def fit(self):
        # implicit likes (items x users)
        self.model.fit(self.train_matrix.T, epochs=self.iterations)

    def recommend(self, user_idx, k=K):
        # model.recommend returns (item_idx, score)
        user_items = self.train_matrix[user_idx]
        recomm = self.model.recommend(user_idx, user_items, N=k, filter_already_liked_items=True)
        items = [i for i, _ in recomm]
        return items


# ----------------------------
# Main orchestration
# ----------------------------
if __name__ == '__main__':
    print("Loading dataset...")
    users, books, ratings = load_kaggle_book_dataset()

    print("Preprocessing and splitting...")
    data = preprocess_and_split(users, books, ratings, test_size=0.2, seed=42)
    train_df = data['train_df']
    test_df = data['test_df']
    train_matrix = data['train_matrix']
    n_users = data['n_users']
    n_items = data['n_items']

    print('\n--- TRAIN / TEST sizes ---')
    print(f"train interactions: {len(train_df)}, test interactions: {len(test_df)}")

    # 1) UserKNN
    print('\nTraining UserKNN...')
    knn = UserKNNRecommender(train_matrix, n_neighbors=20)
    knn_prec, knn_rec, knn_ndcg = evaluate_model(knn.recommend, train_df, test_df, n_users, n_items, k=K)
    print(f"UserKNN -> Precision@{K}: {knn_prec:.4f}, Recall@{K}: {knn_rec:.4f}, NDCG@{K}: {knn_ndcg:.4f}")

    # 2) LightGBM
    print('\nTraining LightGBM...')
    lgbm = LightGBMRecommender(train_df)
    lgbm.fit()
    lgbm_prec, lgbm_rec, lgbm_ndcg = evaluate_model(lgbm.recommend, train_df, test_df, n_users, n_items, k=K)
    print(f"LightGBM -> Precision@{K}: {lgbm_prec:.4f}, Recall@{K}: {lgbm_rec:.4f}, NDCG@{K}: {lgbm_ndcg:.4f}")

    # 3) BPR
    print('\nTraining BPR...')
    bpr = BPRRecommender(train_matrix, factors=64, iterations=50)
    bpr.fit()
    bpr_prec, bpr_rec, bpr_ndcg = evaluate_model(bpr.recommend, train_df, test_df, n_users, n_items, k=K)
    print(f"BPR -> Precision@{K}: {bpr_prec:.4f}, Recall@{K}: {bpr_rec:.4f}, NDCG@{K}: {bpr_ndcg:.4f}")

    print('\nDone. For further experiments, try tuning hyperparameters, adding side-information features for LightGBM, or using more advanced models (DeepFM, DCN).')
