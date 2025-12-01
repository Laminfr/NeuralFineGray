import pandas as pd
import numpy as np

from tabpfn import TabPFNClassifier
from tabpfn_extensions import TabPFNEmbedding

"""
Extracts embeddings from a TabPFN model for downstream tasks.
"""
def get_embeddings_tabpfn(X_train, X_test, t_train, e_train, data_frame_output=True):
    # wrap everything to pandas ---
    X_train = wrap_np_to_pandas(X_train)
    X_test  = wrap_np_to_pandas(X_test)
    t_train = wrap_np_to_pandas(t_train, prefix="t")
    e_train = wrap_np_to_pandas(e_train, prefix="e")

    model = TabPFNClassifier(n_estimators=1)
    embedder = TabPFNEmbedding(tabpfn_clf=model, n_fold=0)
    # pseudo target
    time_bins = pd.qcut(t_train, q=5, labels=False, duplicates='drop')
    y_train = time_bins * 2 + e_train.values
    # get embeddings
    train_emb = embedder.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="train",
    )[0]
    test_emb = embedder.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
    )[0]
    if data_frame_output:
        # Wrap embeddings in DataFrames
        train_embeddings = pd.DataFrame(
            train_emb, columns=[f"x{i}" for i in range(train_emb.shape[1])],
            index=X_train.index
        )
        test_embeddings = pd.DataFrame(
            test_emb, columns=[f"x{i}" for i in range(test_emb.shape[1])],
            index=X_test.index
        )
        return train_embeddings, test_embeddings
    else:
        return train_emb, test_emb
    

def wrap_np_to_pandas(X, index=None, prefix="x"):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X

    X = np.asarray(X)

    if X.ndim == 1:
        return pd.Series(X, index=index)

    if X.ndim == 2:
        n_cols = X.shape[1]
        cols = [f"{prefix}{i}" for i in range(n_cols)]
        return pd.DataFrame(X, columns=cols, index=index)

    raise ValueError("Input must be 1D or 2D numpy array or pandas object.")


