# R3 sources and method provenance

- [Official HistGradientBoostingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html): inspected binary log-loss, early-stopping and explicit X_val/y_val behavior. The installed 1.9.1 signatures were inspected directly. This is the actual reranker implementation.
- [Official LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html): consulted for binary versus ranking objective distinctions. LightGBM was not installed or used; neither a ranking objective nor binary log loss is presented as an exact MAP@5 optimizer.
- Attempts to retrieve the suggested public history-normalization note and rerank2.py pages failed through the web tool. Their contents were not used or claimed to have been read. The rate/share normalization, chronology, features, pool generation and training code were implemented locally from the student's plan and functioning R1/R2 pipeline.

R1's target-source attribution remains in `experiments/EVALUATION_CONTRACT.md` and its sources file. No external predictions, preprocessed data, reports or source code were imported. R2 public/private 0.39904/0.39378 are exact student-reported numbers; original Kaggle evidence was not inspected.
