/*
 * Test gamma-exponential cdf evaluations.
 */
program test_cdf_gamma_exponential(N:Integer <- 10000) {
  m:TestGammaExponential;
  m.initialize();
  handle(m.simulate());
  test_cdf(m.marginal(), N);
}
