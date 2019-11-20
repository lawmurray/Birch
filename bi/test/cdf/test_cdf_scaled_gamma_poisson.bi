/*
 * Test scaled-gamma-Poisson cdf evaluations.
 */
program test_cdf_scaled_gamma_poisson() {
  m:TestScaledGammaPoisson;
  delay.handle(m.simulate());
  test_cdf(m.marginal());
}
