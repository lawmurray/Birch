/*
 * Test gamma-Poisson cdf evaluations.
 */
program test_cdf_gamma_poisson() {
  m:TestGammaPoisson;
  m.play();
  test_cdf(m.marginal());
}
