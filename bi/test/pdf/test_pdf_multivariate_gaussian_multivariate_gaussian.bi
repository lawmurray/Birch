/*
 * Test multivariate Gaussian-Gaussian pdf evaluations.
 */
program test_pdf_multivariate_gaussian_multivariate_gaussian(
    N:Integer <- 10000, B:Integer <- 500, S:Integer <- 20) {
  m:TestMultivariateGaussianMultivariateGaussian;
  playDelay.handle(m.simulate());
  test_pdf(m.marginal(), 5, N, B, S);
}
