/*
 * Test gamma-Poisson cdf evaluations.
 */
program test_cdf_gamma_poisson() {
  m:TestGammaPoisson;
  delay.handle(m.simulate());
  test_cdf(m.marginal());
}
