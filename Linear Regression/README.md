A body fat prediction program that uses multiple regression to fit body fat to other measurements in the given csv file.

1) get_dataset(filename) — takes in a filename and returns the data in an n x (m+1) array.

2) print_stats(dataset, col) — prints several statistics about a given column of the dataset given by get_dataset.

3) regression(dataset, cols, betas) — calculates and returns the mean squared error on the dataset given betas.

4) gradient_descent(dataset, cols, betas) — performs a single step of gradient descent on the MSE and returns the derivative values as an 1D array.

5) iterate_gradient(dataset, cols, betas, T, eta) — performs T iterations of gradient descent starting at the given betas and prints the results.

6) compute_betas(dataset, cols) — calculates and returns the values of betas and the corresponding MSE as a tuple.

7) predict(dataset, cols, features) — returns the predicted body fat percentage of the given features.

8) synthetic_datasets(betas, alphas, X, sigma) — generates two synthetic datasets, one using a linear model and the other using a quadratic model.

9) plot_mse() — fits the synthetic datasets, and plots a figure depicting the MSEs under different situations.