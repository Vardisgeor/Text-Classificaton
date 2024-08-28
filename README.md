# Text Classification with Machine Learning Algorithms

This repository contains the implementation and comparison of various machine learning algorithms for text classification, specifically for classifying IMDB movie reviews as positive or negative. The project includes custom implementations of AdaBoost and Random Forest algorithms, as well as a Multi-Layer Perceptron (MLP) model.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms Implemented](#algorithms-implemented)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project aims to explore the effectiveness of different machine learning algorithms for text classification. It involves the implementation of AdaBoost and Random Forest from scratch, followed by a comparison with their scikit-learn counterparts. Additionally, an MLP model is implemented to compare performance with traditional methods.

## Algorithms Implemented

### 1. AdaBoost
- Custom implementation using weak learners (Decision Trees).
- Adjusts weights of training data iteratively to improve model accuracy.

### 2. Random Forest
- Custom implementation based on the ID3 decision tree algorithm.
- Combines multiple decision trees to make final predictions.

### 3. MLP (Multi-Layer Perceptron)
- A neural network model implemented to classify text data.
- Capable of learning complex non-linear functions for better accuracy.

## Dataset

The dataset used is the IMDB movie reviews dataset. It includes movie descriptions that are classified as either positive or negative.

## Results

The results show a comparison between custom implementations and scikit-learn implementations of AdaBoost and Random Forest, as well as an MLP model. The MLP model outperforms the other models in terms of accuracy and execution time.

### Accuracy Comparison

| Model                      | Accuracy (5000 samples) |
|----------------------------|------------------------|
| Custom AdaBoost            | 75.12%                 |
| scikit-learn AdaBoost      | 79.48%                 |
| Custom Random Forest       | 71.59%                 |
| scikit-learn Random Forest | 72.50%                 |
| MLP                        | Best performance       |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
