/*
 * Test inverse-gamma-gamma cdf evaluations.
 */
program test_cdf_inverse_gamma_gamma(N:Integer <- 10000) {
  m:TestInverseGammaGamma;
  delay.handle(m.simulate());
  test_cdf(m.marginal(), N);
}
