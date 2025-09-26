# CS5710 Machine Learning - Homework 2 Solutions

This repository contains solutions to Homework 2 for **CS5710 Machine Learning**.  
It includes both **calculation-based answers (Part A)** and **Python programming tasks (Part B)**.

---
##  Student Info
- Name: M.Siva sai kumar
- Student ID : 700773075
- Course: CS5710 Machine Learning
- Semester: Fall 2025

---

## Contents
- `Homework_2_Solutions.docx` → Full written solutions to Q1–Q6.
- `DecisionTree.py` → Code for Q7 (Decision Tree on Iris dataset).
- `KNN_Classification.py` → Code for Q8 (kNN classification with decision boundaries).
- `Performance_Evaluation.py` → Code for Q9 (kNN evaluation: confusion matrix, metrics, ROC/AUC).
- `README.md` → This documentation.

---

##  Part A: Calculation Questions (Q1–Q6)
Solved step-by-step in the Word file. Covers:
- Decision stump prediction
- Training error as splitting criterion
- Entropy & information gain
- Confusion matrix metrics
- Distance calculations (kNN)
- k-fold cross-validation

---

##  Part B: Programming Questions (Q7–Q9)

### Q7. Decision Tree (sklearn)
Trains Decision Trees with `max_depth = 1, 2, 3` on the **Iris dataset**.  
Reports training & test accuracies and discusses underfitting/overfitting.

Run:
```bash
python DecisionTree.py
```

---

### Q8. kNN Classification (sklearn)
Uses only **sepal length** and **sepal width** features of the Iris dataset.  
Trains kNN classifiers with `k=1, 3, 5, 10` and plots decision boundaries.

Run:
```bash
python KNN_Classification.py
```

---

### Q9. Performance Evaluation (kNN, k=5)
Trains a kNN classifier (`k=5`) on the Iris dataset.  
Outputs:
- Confusion Matrix
- Accuracy, Precision, Recall, F1 (classification_report)
- ROC curve & AUC for each class

Run:
```bash
python Performance_Evaluation.py
```

---

##  Installation

Before running the scripts, install required packages:

```bash
pip install scikit-learn matplotlib numpy
```

---





