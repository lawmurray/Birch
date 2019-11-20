/*
 * Test normal-inverse-gamma Gaussian cdf evaluations.
 */
program test_cdf_normal_inverse_gamma_gaussian(N:Integer <- 10000) {
  m:TestNormalInverseGammaGaussian;
  delay.handle(m.simulate());
  test_cdf(m.marginal(), N);
}
