/*
 * Test beta-Bernoulli pdf evaluations.
 */
program test_pdf_beta_bernoulli(N:Integer <- 10000) {
  m:TestBetaBernoulli;
  delay.handle(m.simulate());
  test_pdf(m.marginal(), N);
}
