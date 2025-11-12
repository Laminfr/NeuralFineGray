import pandas as pd
from tarte_ai import TARTE_TableEncoder, TARTE_TablePreprocessor
from sklearn.pipeline import Pipeline

"""
Extracts embeddings from a TARTE model for downstream tasks.
"""
def get_embeddings_tarte(X_train, X_test, t_train, e_train):
    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    # pseudo target
    time_bins = pd.qcut(t_train, q=5, labels=False, duplicates='drop')
    y_train = time_bins * 2 + e_train.values
    # get embeddings
    train_emb = prep_pipe.fit_transform(X_train, y_train)
    test_emb = prep_pipe.transform(X_test)
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