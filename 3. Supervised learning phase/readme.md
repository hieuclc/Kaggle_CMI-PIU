# 3. Supervised learning phase

Section 3 comprises two notebooks, referred to as following:
* 3.1 Supervised learning
* 3.2 VIME & Supervised learning

The first notebook (3.1) consists of two phases. The first phase involves loading and processing the data, while the second phase utilizes supervised learning on the labeled dataset.

The second notebook (3.2) includes three phases. The first phase is also to loading and processing the data. In the second phase, VIME is employed to extract representative information from the tabular data, which is then used to generate labels for the unlabeled samples. The final phase of this notebook applies supervised learning to the entire labeled dataset.

Both notebooks share the common first phase of loading and processing data and conclude with a supervised learning phase.

In the supervised learning phase, we build three models that are popular and effective for classification problems often encountered in Kaggle competitions: LightGBM, XGBoost, and CatBoost. The hyperparameters for all three models are tuned using the Randomized Grid Search Cross-Validation technique ([For more detail](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)). Ultimately, these models are combined using the ensemble technique known as Voting Regressor.