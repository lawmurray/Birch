/*
 * Test linear multivariate normal-inverse-gamma Gaussian pdf evaluations.
 */
program test_pdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
    D:Integer <- 4, N:Integer <- 10000, B:Integer <- 500, S:Integer <- 20) {
  m:TestLinearMultivariateNormalInverseGammaMultivariateGaussian;
  m.play();
  test_pdf(m.marginal(), 5, N, B, S);
}
