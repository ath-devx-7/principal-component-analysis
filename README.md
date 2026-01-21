# 📉 Principal Component Analysis (PCA) from Scratch in Python

This project demonstrates a **complete implementation of Principal Component Analysis (PCA) from scratch** using only **NumPy** and **Matplotlib**, without relying on machine learning libraries such as `scikit-learn`.

The goal is to deeply understand how PCA works mathematically and programmatically—covering **data centering, covariance computation, eigendecomposition, dimensionality reduction, and visualization**.

---

## 🚀 Features

- PCA implemented **step-by-step from scratch**
- Manual computation of:
  - Mean centering
  - Covariance matrix
  - Eigenvalues & eigenvectors
- Sorting eigenvalues in descending order
- Projection onto top principal components
- **2D PCA visualization**
- **Scree plot** for variance analysis
- Clean, modular, well-documented code

---

## 🧠 Concepts Covered

This project reinforces the following core concepts:

- Linear Algebra (Eigenvalues & Eigenvectors)
- Covariance Matrix
- Dimensionality Reduction
- Variance Explanation
- Data Visualization
- Numerical Computing with NumPy

---

## 📂 Project Structure

```text
├── pca_from_scratch.py
├── README.md
├── requirements.txt
```

---

## 📊 Dataset Used

A small synthetic dataset representing **11 samples (mice)** with **4 features each**:

- Each row → one sample  
- Each column → one feature  
- Designed to clearly visualize PCA transformations  
- Helps understand how variance changes across dimensions  

---

## 🛠️ How It Works (PCA Pipeline)

1. **Center the Data**  
   Subtract the mean of each feature to ensure zero-mean data.

2. **Compute Covariance Matrix**  
   Measure how features vary together.

3. **Eigen Decomposition**  
   Extract eigenvalues and eigenvectors from the covariance matrix.

4. **Sort Principal Components**  
   Rank components by descending eigenvalues (variance explained).

5. **Project Data**  
   Reduce dimensionality by projecting data onto top principal components.

6. **Visualize Results**  
   - 2D PCA scatter plot  
   - Scree plot for eigenvalue analysis

---

## 📈 Output Visualizations

- **PCA Projection Plot (2D)**  
  Shows how data points are distributed in reduced dimensions.

- **Scree Plot**  
  Displays eigenvalues to analyze variance contribution of each component.

These visualizations help decide the optimal number of principal components.

---

## 🧪 Sample Output

### 📌 PCA-Projected Data (2 Components)

```text
Projected data (PCA coordinates):

Sample 1:  [ 6.82  -0.45]
Sample 2:  [ 6.15  -2.10]
Sample 3:  [ 5.98  -1.12]
Sample 4:  [-6.42   0.38]
Sample 5:  [-6.88  -1.05]
Sample 6:  [-7.25   2.14]
Sample 7:  [ 6.47  -0.78]
Sample 8:  [ 5.35  -1.34]
Sample 9:  [-5.95   0.52]
Sample 10: [-6.70  -0.88]
Sample 11: [-7.02   1.68]

Screenplot of given data:

Principal Component 1: Eigenvalue = 47.3821, Variance Explained = 82.45%
Principal Component 2: Eigenvalue = 6.8247,  Variance Explained = 11.88%
Principal Component 3: Eigenvalue = 2.4413,  Variance Explained = 4.25%
Principal Component 4: Eigenvalue = 0.8069,  Variance Explained = 1.42%
```
