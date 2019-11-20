/*
 * Test beta-binomial cdf evaluations.
 */
program test_cdf_beta_binomial() {
  m:TestBetaBinomial;
  delay.handle(m.simulate());
  test_cdf(m.marginal());
}
