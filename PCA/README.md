A collection of functions for a facial Analysis program that uses PCA (Principal Component Analysis).

1) load_and_center_dataset(filename) - loads the dataset from the given .npy file, centers it around the origin, and returns it as a NumPy array of floats.

2) get_covariance(dataset) — calculates and return the covariance matrix of the dataset as a d x d NumPy matrix.

3) get_eig(S, m) — performs eigen decomposition on the covariance matrix S and returns a diagonal matrix with the largest m eigenvalues on the diagonal, and a matrix with the corresponding eigenvectors as columns.

4) get_eig_perc(S, perc) — Instead of returning the first m eigenvalues and corresponding eigenvectors, returns all that explain more than a certain % of variance.

5) project_image(image, U) — projects each image into the m-dimensional space and returns the new representation as a d x 1 array.

6) display_image(orig, proj) — uses matplotlib to display the original image and the projected image side-by-side.