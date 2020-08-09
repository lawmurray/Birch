/*
 * Test Gaussian-Gaussian conjugacy.
 */
program test_gaussian_gaussian(N:Integer <- 10000) {
  m:TestGaussianGaussian;
  test_conjugacy(m, N, 2);
}
