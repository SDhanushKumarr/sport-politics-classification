import os
import re
import random
import math
from collections import Counter, defaultdict

# -----------------------------
# Data loading and preprocessing
# -----------------------------

def load_bbc_sport_politics(base_dir="data/bbc"):
    data = []
    for label in ["sport", "politics"]:
        folder = os.path.join(base_dir, label)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r", encoding="latin-1") as f:
                text = f.read().strip()
                if text:
                    data.append((text, label))
    random.shuffle(data)
    return data

def tokenize(text):
    # simple tokenization: lowercase, split into word characters
    return re.findall(r"\w+", text.lower())

def train_test_split(data, test_ratio=0.2):
    n = len(data)
    test_size = int(n * test_ratio)
    test = data[:test_size]
    train = data[test_size:]
    return train, test

# -----------------------------
# Feature extraction
# -----------------------------

def build_vocab(train_data, max_vocab_size=10000, min_freq=2):
    freq = Counter()
    for text, label in train_data:
        tokens = tokenize(text)
        freq.update(tokens)
    items = [(w, c) for w, c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])
    items = items[:max_vocab_size]
    vocab = {w: i for i, (w, c) in enumerate(items)}
    return vocab

def vectorize_bow(tokens, vocab):
    vec = Counter()
    for t in tokens:
        if t in vocab:
            idx = vocab[t]
            vec[idx] += 1
    return vec

def compute_idf(train_data, vocab):
    df = defaultdict(int)
    total_docs = len(train_data)
    for text, label in train_data:
        tokens = tokenize(text)
        seen = set()
        for t in tokens:
            if t in vocab and t not in seen:
                df[t] += 1
                seen.add(t)
    idf = {}
    for w, idx in vocab.items():
        df_w = df.get(w, 0) + 1  # smoothing
        idf[w] = math.log(total_docs / df_w)
    return idf

def vectorize_tfidf(tokens, vocab, idf):
    tf = Counter()
    for t in tokens:
        if t in vocab:
            tf[t] += 1
    vec = {}
    for w, count in tf.items():
        idx = vocab[w]
        vec[idx] = count * idf[w]
    return vec

# -----------------------------
# Multinomial Naive Bayes
# -----------------------------

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_log_probs = {}
        self.vocab_size = 0

    def fit(self, X, y):
        labels = list(set(y))
        self.vocab_size = 0
        for vec in X:
            if vec:
                self.vocab_size = max(self.vocab_size, max(vec.keys()) + 1)

        word_counts = {c: Counter() for c in labels}
        class_counts = Counter(y)

        for vec, label in zip(X, y):
            word_counts[label].update(vec)

        total_docs = len(y)
        for c in labels:
            self.class_priors[c] = math.log(class_counts[c] / total_docs)

        self.feature_log_probs = {}
        for c in labels:
            total_words_c = sum(word_counts[c].values())
            denom = total_words_c + self.alpha * self.vocab_size
            probs = {}
            for i in range(self.vocab_size):
                count = word_counts[c][i]
                probs[i] = math.log((count + self.alpha) / denom)
            self.feature_log_probs[c] = probs

    def predict_one(self, vec):
        best_label = None
        best_log_prob = -1e18
        for c in self.class_priors:
            log_prob = self.class_priors[c]
            feats = self.feature_log_probs[c]
            for i, count in vec.items():
                if i in feats:
                    log_prob += count * feats[i]
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_label = c
        return best_label

    def predict(self, X):
        return [self.predict_one(vec) for vec in X]

# -----------------------------
# Perceptron (linear classifier)
# -----------------------------

class Perceptron:
    def __init__(self, epochs=5, lr=1.0):
        self.epochs = epochs
        self.lr = lr
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        label_map = {"sport": 1, "politics": -1}
        n_features = 0
        for vec in X:
            if vec:
                n_features = max(n_features, max(vec.keys()) + 1)
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for vec, label in zip(X, y):
                y_true = label_map[label]
                score = self.bias
                for i, val in vec.items():
                    if i < len(self.weights):
                        score += self.weights[i] * val
                y_pred = 1 if score >= 0 else -1
                if y_pred != y_true:
                    for i, val in vec.items():
                        if i < len(self.weights):
                            self.weights[i] += self.lr * y_true * val
                    self.bias += self.lr * y_true

    def predict_one(self, vec):
        score = self.bias
        for i, val in vec.items():
            if i < len(self.weights):
                score += self.weights[i] * val
        return "sport" if score >= 0 else "politics"

    def predict(self, X):
        return [self.predict_one(vec) for vec in X]

# -----------------------------
# Evaluation
# -----------------------------

def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if y_true else 0.0

# -----------------------------
# Main pipeline
# -----------------------------

def main():
    random.seed(42)

    data = load_bbc_sport_politics()
    print("Total documents:", len(data))

    train_data, test_data = train_test_split(data, test_ratio=0.2)
    print("Train size:", len(train_data), "Test size:", len(test_data))

    vocab = build_vocab(train_data, max_vocab_size=10000, min_freq=2)
    print("Vocab size:", len(vocab))

    X_train_bow, y_train = [], []
    for text, label in train_data:
        X_train_bow.append(vectorize_bow(tokenize(text), vocab))
        y_train.append(label)

    X_test_bow, y_test = [], []
    for text, label in test_data:
        X_test_bow.append(vectorize_bow(tokenize(text), vocab))
        y_test.append(label)

    idf = compute_idf(train_data, vocab)
    X_train_tfidf = [vectorize_tfidf(tokenize(t), vocab, idf) for t, _ in train_data]
    X_test_tfidf = [vectorize_tfidf(tokenize(t), vocab, idf) for t, _ in test_data]

    # Model 1: Naive Bayes + BoW
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train_bow, y_train)
    pred_nb = nb.predict(X_test_bow)
    acc_nb = accuracy(y_test, pred_nb)
    print("Accuracy NB+BoW:", acc_nb)

    # Model 2: Perceptron + BoW
    perc_bow = Perceptron(epochs=5, lr=1.0)
    perc_bow.fit(X_train_bow, y_train)
    pred_perc_bow = perc_bow.predict(X_test_bow)
    acc_perc_bow = accuracy(y_test, pred_perc_bow)
    print("Accuracy Perceptron+BoW:", acc_perc_bow)

    # Model 3: Perceptron + TF-IDF
    perc_tfidf = Perceptron(epochs=5, lr=1.0)
    perc_tfidf.fit(X_train_tfidf, y_train)
    pred_perc_tfidf = perc_tfidf.predict(X_test_tfidf)
    acc_perc_tfidf = accuracy(y_test, pred_perc_tfidf)
    print("Accuracy Perceptron+TFIDF:", acc_perc_tfidf)

if __name__ == "__main__":
    main()
