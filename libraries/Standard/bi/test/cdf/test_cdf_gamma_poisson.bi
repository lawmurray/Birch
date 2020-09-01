/*
 * Test gamma-Poisson cdf evaluations.
 */
program test_cdf_gamma_poisson(N:Integer <- 10000) {
  m:TestGammaPoisson;
  m.initialize();
  handle(m.simulate());
  test_cdf(m.marginal());
}
