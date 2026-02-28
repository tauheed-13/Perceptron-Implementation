# Perceptron-Implementation
A NumPy-based implementation of the Perceptron algorithm from scratch

A clean and minimal implementation of the Perceptron Algorithm built completely from scratch using only NumPy.

This project demonstrates the mathematical foundations of linear classification without using any machine learning libraries like scikit-learn.

## Features
- Binary classification
- Labels supported: -1 and +1 or  0 and 1
- Custom learning rate
- Custom number of epochs
- Fully vector-based implementation using NumPy
- Educational and beginner-friendly
- Uses Perceptron Loss and Hinge Loss both

## Project Structure
.
├── perceptron.py  // Contains basic mathematical impmentation using 0 and 1 as class labels
├── perceptron2.py // Contains implementation using Perceptron Loss and -1 and 1 as class labels
├── perceptron3.py // Contains implementation using Hinge Loss and -1 and 1 as class labels
└── README.md

## Difference: Perceptron Loss vs Hinge Loss:
Perceptron Loss:    L=max(0,−yz)
Hinge Loss:       L=max(0, 1-yz)
Even if prediction is correct but close to boundary, hinge loss still updates weights, but Perceptron Loss only updates weights when prediction is wrong

Hinge loss is needed because the perceptron only cares about whether a point is classified correctly or not, but it does not care how confidently it is classified. Suppose we have two positive points. One is far away from the decision boundary and clearly positive. The other is just barely on the positive side of the boundary. In perceptron, both are treated the same because both are “correct.” The model will not update weights for either of them. But this is risky: if the boundary slightly shifts due to noise, the second point may easily become misclassified. Hinge loss (used in SVM) solves this by introducing the idea of a margin. It says: a prediction is not good enough just because it is correct — it must also be confidently correct. If a point lies too close to the boundary (i.e. yz<1), hinge loss still gives a penalty and updates the weights. This pushes the decision boundary further away from data points, creating a larger safety gap (margin) between classes. Because of this margin maximization, hinge loss gives better generalization and more stable convergence compared to perceptron, especially when data is noisy or not perfectly separable.
