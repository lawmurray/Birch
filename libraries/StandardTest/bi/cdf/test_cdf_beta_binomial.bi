/*
 * Test beta-binomial cdf evaluations.
 */
program test_cdf_beta_binomial(N:Integer <- 10000) {
  m:TestBetaBinomial;
  m.initialize();
  handle(m.simulate());
  test_cdf(m.marginal());
}
