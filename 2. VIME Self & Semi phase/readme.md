# 2. VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain

As outlined in our report, we are pursuing two distinct approaches:
* The first approach employs only Supervised Learning on the labeled dataset, which comprises two-thirds of the entire dataset. 
* The second one combines Semi-Supervised and Self-Supervised Learning, allowing us to fully leverage the entire dataset, including both labeled and unlabeled data. This enables us to label any missing data and subsequently apply Supervised Learning to this enriched dataset.

In this section, we focus on implementing [VIME](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf), a model proposed in 2020 at the NeurIPS conference by four researchers: Jinsung Yoon, Yao Zhang, James Jordon, and Mihaela van der Schaar. VIME integrates Self-Supervised and Semi-Supervised Learning techniques with the goal of extending their effectiveness from image and text data to tabular data. 

The code associated with this paper can be found at the following GitHub link [VIME Implementation](https://github.com/jsyoon0823/VIME). However, due to the use of outdated architectures, much of it is not usable. We have re-implemented using the modern TensorFlow 2.x architecture, which is compatible with our problem and works effectively.

In addition to re-implementing using TensorFlow 2.x architecture, we also modified the supervised loss function to better suit the ordinal classification problem of the competition. The Weighted Kappa loss function, introduced in 2018 by researchers Jordi de la Torre, Domenec Puig, and Aida Valls, is specifically designed for ordinal data ([More details](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666)).



