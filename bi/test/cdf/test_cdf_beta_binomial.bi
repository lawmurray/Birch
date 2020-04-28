/*
 * Test beta-binomial cdf evaluations.
 */
program test_cdf_beta_binomial() {
  m:TestBetaBinomial;
  m.initialize();
  handle(m.simulate());
  test_cdf(m.marginal());
}
