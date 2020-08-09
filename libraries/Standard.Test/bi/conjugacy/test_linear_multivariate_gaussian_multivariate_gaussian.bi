/*
 * Test multivariate linear Gaussian-Gaussian conjugacy.
 */
program test_linear_multivariate_gaussian_multivariate_gaussian(
    N:Integer <- 10000) {
  m:TestLinearMultivariateGaussianMultivariateGaussian;
  test_conjugacy(m, N, 10);
}
