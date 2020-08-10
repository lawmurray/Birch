/*
 * Test scaled-gamma-Poisson cdf evaluations.
 */
program test_cdf_scaled_gamma_poisson(N:Integer <- 10000) {
  m:TestScaledGammaPoisson;
  m.initialize();
  handle(m.simulate());
  test_cdf(m.marginal());
}
