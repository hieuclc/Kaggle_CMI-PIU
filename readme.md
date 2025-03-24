# Kaggle Code Competition: Child Mind Institute — Problematic Internet Use

## Overview
The [Kaggle competition: Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) was hosted on the Kaggle platform, sponsored by Dell & NVIDIA. Our team achieved the *Kaggle Silver Medal*, ranked 94th out of 3,559 teams. The repository includes different implementation approaches and the final-term report for the course *Introduction to Machine Learning* at the University of Engineering and Technology, Vietnam National University, Hanoi.

## Contributors
- [Nguyễn Hữu Thế - 22028155](https://github.com/thebeo2004)
- [Ngô Duy Hiếu - 22028280](https://github.com/hieuclc)
- [Nguyễn Hữu Tiến - 22028180](https://github.com/tien1712)

## Project structure
```text
Kaggle_CMI-PIU/
│
├── data/                  # Data files (gitignored, which is freely accessed during the competition)
│   ├── train.csv
│   ├── test.csv
│   ├── data_dictionary.csv
│   └── sample_submission.csv
│
├── 1. Data loading and processing/
│   └── 1.1 Tabular data analysis.ipynb
│
├── 2. Semi-supervised learning phase/
│   └── [Notebooks for semi-supervised learning]
│
├── 3. Supervised learning phase/
│   ├── 3.1 Supervised learning.ipynb
│   ├── 3.2 VIME & Supervised learning.ipynb
│   └── readme.md
│
└── [Other project files]
```

## Methodology
![Our proposed methodology](./Methodology.png)

The competition introduced the Healthy Brain Network dataset, which includes clinical and research data of around five-thousand participants aged 5-22. The dataset is considered fairly challenging as it features a mix of time-series actigraphy data and tabular clinical assessments, while also presenting inconsistencies in the given physical activity data and Internet usage behavior data. Our solution applies semi-supervised and supervised learning techniques to effectively utilize the given dataset. With this hybrid approach, we aim to overcome data limitations and modelize associations between Internet usage patterns, physical activity and clinical measures.

Key components of our methodology:
### 1. Data Loading & Processing
- Analyzing demographic, physical, and bioelectrical impedance measurements
- Addressing missing values using different imputation techniques
- Feature engineering to derive meaningful health-related metrics
### 2. Semi-supervised Learning Approach
- Implementing the VIME (Value Imputation and Mask Estimation) framework
- Extracting feature representations from unlabeled data
- Creating high-confidence pseudo-labels for previously unlabeled samples
### 3. Supervised Learning Approach
- Training an ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost)
- Optimizing hyperparameters with Randomized Grid Search CV
- Model stacking with Voting Regressor to leverage different algorithm strengths

## Results & Conclusions
![Result table](./result.png)

The competition used *public* and *private* dataset Quadratic Weighted Kappa (QWK) scores (as shown in the table) for evaluation, with the final ranking based on the *private* dataset QWK score. Our team evaluated the approaches of using only labeled samples from the original dataset ("**Only labeled data**") and applying VIME to generate better-quality labeled samples ("**Entire dataset**"). While VIME produced remarkable results on well-established datasets such as MNIST (converted to tabular format), UCI Income, UCI Blog, and genetic datasets related to blood cell characteristics (MRV, MPV, MCH, RET, PCT, MONO), it is still highly dependent on good pretraining data, making its performance vulnerable to the weaknesses of the given competition dataset, especially when working with unseen data.

The final ranking suggests the importance of more focused efforts on data cleaning, preprocessing, and imputation techniques to improve model performance.

For more detail, refer to the [CMI-PIU Competition Report](./report.pdf).

## References

- [VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf) - Yoon, J., Zhang, Y., Jordon, J., and van der Schaar, M. (2020). Advances in Neural Information Processing Systems, 33, 11033-11043.

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) - Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). Proceedings of the 31st International Conference on Neural Information Processing Systems, 3149-3157.

- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) - Chen, T., and Guestrin, C. (2016). Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.

- [CatBoost: Unbiased Boosting with Categorical Features](https://papers.nips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) - Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A., and Gulin, A. (2018). Advances in Neural Information Processing Systems, 31.
