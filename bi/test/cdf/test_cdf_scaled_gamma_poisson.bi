/*
 * Test scaled-gamma-Poisson cdf evaluations.
 */
program test_cdf_scaled_gamma_poisson() {
  m:TestScaledGammaPoisson;
  m.play();
  test_cdf(m.marginal());
}
