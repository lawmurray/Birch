/*
 * Test linear multivariate Gaussian-Gaussian pdf evaluations.
 */
program test_pdf_linear_multivariate_gaussian_multivariate_gaussian(
    D:Integer <- 4, N:Integer <- 20000, B:Integer <- 100, S:Integer <- 20) {
  m:TestLinearMultivariateGaussianMultivariateGaussian;
  m.play();
  test_pdf(m.marginal(), 5, N, B, S);
}
