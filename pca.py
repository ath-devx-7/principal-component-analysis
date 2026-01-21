import numpy as np
import matplotlib.pyplot as plt

def centre_data(X):
    """
    Centers the data by subtracting the mean of each feature.
    """
    mean = np.mean(X, axis=0)
    return X - mean

def compute_covariance_matrix(X):
    """
    Computes the covariance matrix of the centred data.
    """
    return np.cov(X, rowvar=False)

def eigendecomposition(cov_matrix):
    """
    Performs eigendecomposition on the covariance matrix
    and sorts eigenvalues & eigenvectors in descending order.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

def project_data(X, eigenvectors, num_components=2):
    """
    Projects data onto the top principal components.
    """
    top_eigenvectors = eigenvectors[:, :num_components]
    return X @ top_eigenvectors

def perform_pca(X, num_components=2):
    """
    Complete PCA pipeline.
    """
    centred_X = centre_data(X)
    cov_matrix = compute_covariance_matrix(centred_X)
    eigenvalues, eigenvectors = eigendecomposition(cov_matrix)
    projected_X = project_data(centred_X, eigenvectors, num_components)

    return projected_X, eigenvalues, eigenvectors

def print_projected_data(projected_X):
    """
    Prints PCA-projected data in the terminal.
    """
    print("\nProjected data (PCA coordinates):")

    for i, point in enumerate(projected_X):
        print(f"Sample {i+1}: {point}")

def plot_projected_data(projected_X):
    """
    Visualizes PCA-projected data (2D).
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(projected_X[:, 0], projected_X[:, 1])

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection (2 Components)")

    plt.grid(True)
    plt.show()

def print_scree_plot(eigenvalues):
    """
    Calculate Screenplot data of all eigenvalues.
    """
    sum_eigenvalues = np.sum(eigenvalues)
    percentage_variance = [(val / sum_eigenvalues) * 100 for val in eigenvalues]
    print("\nScreenplot of given data:")
    for i in range(len(eigenvalues)):
        print(f"Principal Component {i+1}: Eigenvalue = {eigenvalues[i]:.4f}, Variance Explained = {percentage_variance[i]:.2f}%")

def plot_scree(eigenvalues):
    plt.figure()
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.show()


X = np.array([
    [10,  6, 12, 5],    # Mouse 1
    [11,  4,  9, 7],    # Mouse 2
    [ 8,  5, 10, 6],    # Mouse 3
    [ 3,  3,  2.5, 2],  # Mouse 4
    [ 2,  2.8, 1.3, 4], # Mouse 5
    [ 1,  1,  2, 7],     # Mouse 6
    [9,   5.5, 11, 5.5],   # Mouse 7
    [7.5, 4.8, 9.2, 6.2],  # Mouse 8
    [3.5, 3.2, 3.0, 2.5],  # Mouse 9
    [2.2, 2.5, 1.8, 3.8],  # Mouse 10
    [1.5, 1.2, 2.5, 6.5]   # Mouse 11
])

projected_X, eigenvalues, eigenvectors = perform_pca(X, num_components=2)

print_projected_data(projected_X)
print_scree_plot(eigenvalues)
plot_projected_data(projected_X)
plot_scree(eigenvalues)