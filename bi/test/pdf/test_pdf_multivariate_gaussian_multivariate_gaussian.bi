/*
 * Test multivariate Gaussian-Gaussian pdf evaluations.
 */
program test_pdf_multivariate_gaussian_multivariate_gaussian(D:Integer <- 4,
    N:Integer <- 10000, B:Integer <- 10000, S:Integer <- 10) {
  m:TestMultivariateGaussianMultivariateGaussian;
  m.play();
  test_pdf(m.marginal(), 5, N, B, S);
}
