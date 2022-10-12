# Credit Default Prediction

Predicting credit card default is an important task for payment card providers to manage risk and take appropriate preventative action. It is often claimed that the tabular structure of performance indicators related to these events remains a unique challenge to deep learning based methods, despite their numerous breakthroughs in other domains. In this work, we aim to verify this claim by training and systematically analyzing several deep learning architectures for the task of credit default detection and subsequently comparing the results to a carefully tuned decision tree based classifier. The best performing deep learning architecture achieves a final evaluation score of 0.783, whereas the decision tree based model achieves 0.791. This result confirms prior work in the area of credit default prediction, stating that decision tree based models reach state-of-the-art performance that deep learning based method fail to surpass.

See the [report](./Report/Advanced_Machine_Learning_SoSe_22.pdf) for more information.

This repository contains accompanying code for the Credit Default Prediction project.

A quick overview:

- `main.py` and associated imports contains Python code related to Hyperparameter Optimization of preprocessing pipeline (Sec. 3.4)
- `explore_*.py` contains various one-off tasks, experiments, exploratory data analysis
- `explore_train_catboost_final.py` contains Catboost ensemble training detailled in Sec. 3.5 and 4.3
- `model_*.ipynb` contains training code for DL approaches
- `R/` contains R code with further analysis (method ranking analysis, box-and-dotplots, Wilcoxon signed rank test).
