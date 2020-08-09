/*
 * Test multivariate Gaussian-Gaussian conjugacy.
 */
program test_multivariate_gaussian_multivariate_gaussian(N:Integer <- 10000) {
  m:TestMultivariateGaussianMultivariateGaussian;
  test_conjugacy(m, N, 10);
 }
