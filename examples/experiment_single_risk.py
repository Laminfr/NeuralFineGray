# Comparsion models for single risks

from experiment import TabPFNExperiment

random_seed = 0

# test dataset breast cancer, , replace with our datasets
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# test estimator, replace with our baselines
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression(solver='lbfgs', max_iter=5000)

# TabPFN experiment
exp_1 = TabPFNExperiment(X, y, estimator, use_embeddings=True, test_size=0.5)
exp_1.fit()
y_pred_emb = exp_1.predict()
exp_2 = TabPFNExperiment(X, y, estimator, use_embeddings=False, test_size=0.5)
exp_2.fit()
y_pred_raw = exp_2.predict()

# accuracy
from sklearn.metrics import accuracy_score
acc_emb = accuracy_score(exp_1.y_test, y_pred_emb)
print("accuracy score using TabPFN embeddings + LogisticRegression:", "{:.4f}".format(acc_emb))
acc_raw = accuracy_score(exp_2.y_test, y_pred_raw)
print("accuracy score using raw data + LogisticRegression:", "{:.4f}".format(acc_raw))