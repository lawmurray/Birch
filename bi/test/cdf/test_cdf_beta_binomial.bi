/*
 * Test beta-binomial cdf evaluations.
 */
program test_cdf_beta_binomial() {
  m:TestBetaBinomial;
  m.play();
  test_cdf(m.marginal());
}
