🧠 Hotel Review Score Prediction — BiLSTM Model
Overview

This project was developed by Marco Sabbadin (Matricola: 536040) as part of the Machine Learning, Artificial Neural Networks, and Deep Learning exam at Università degli Studi di Pavia.

The goal is to predict hotel review scores based on review text using a Bidirectional LSTM (BiLSTM) neural network.
The model processes text reviews, learns contextual patterns, and estimates the corresponding numeric review scores.

🎯 Objectives

Predict continuous Review_Score values from textual hotel reviews.

Preprocess text data to create clean, numerical sequences suitable for neural networks.

Design and train a BiLSTM model with embedding and regularization.

Perform a Grid Search over activation functions and weight initializers.

Evaluate model performance on unseen data using robust metrics.

🧩 Dataset

Source: GitHub-hosted CSV

Columns used:

Review – textual hotel review

Review_Score – numeric score assigned by the reviewer

After cleaning and dropping missing values, the dataset is split into:

Training Set: 64%

Validation Set: 16%

Test Set: 20%

⚙️ Data Preprocessing

Text preprocessing includes:

Removing hyphens, punctuation, and non-alphabetic characters

Lowercasing and tokenization

Converting words to integer indices

Padding sequences to a fixed length (MAX_LEN = 150)

Each review is represented as a numerical vector of word indices, ready for embedding.

🧠 Model Architecture

The neural network follows this structure:

Layer	Description
Embedding	Transforms words into dense 64-dimensional vectors
Bidirectional LSTM (64 units)	Captures forward and backward text dependencies
Dropout (0.4)	Prevents overfitting
Dense (1 neuron)	Outputs the predicted review score

Additional configurations:

Loss: Mean Squared Error (MSE)

Optimizer: Adam

Regularization: L2 (λ = 1e−4)

Early Stopping: patience = 2

Epochs: up to 20

🔍 Grid Search

A simplified Grid Search is performed to compare two activation functions and their corresponding initializers:

Activation	Weight Initializer	Description
relu	HeNormal	Standard for ReLU layers
linear	GlorotUniform	Suitable for regression tasks

The model is trained for 5 epochs per configuration to identify the best-performing setup within execution limits.

📈 Evaluation

Model performance is assessed on the test set using:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Early stopping ensures generalization and avoids overfitting.
The evaluation demonstrates that the BiLSTM effectively captures text semantics relevant to the scoring task.

🧮 Technologies

Language: Python 3.10

Frameworks: TensorFlow / Keras

Libraries:

pandas, numpy, scikit-learn — data manipulation and splitting

matplotlib — visualization

tensorflow.keras — deep learning model and training

🚀 How to Run

Clone the repository and install dependencies:

pip install tensorflow pandas numpy scikit-learn matplotlib


Open the notebook:

jupyter notebook Sabbadin_536040_exam.ipynb


Run all cells sequentially to:

Load and preprocess data

Train models with grid search

Evaluate and visualize performance

🧠 Insights

Text preprocessing quality heavily affects performance.

BiLSTM architecture captures both contextual directions, improving regression accuracy.

Regularization and early stopping effectively prevent overfitting.

The best configuration uses an activation function aligned with the regression objective.

👨‍💻 Author

Marco Sabbadin
Bachelor’s Degree in Artificial Intelligence
Università degli Studi di Pavia
