/*
 * Test beta-Bernoulli pdf evaluations.
 */
program test_pdf_beta_bernoulli(N:Integer <- 10000) {
  m:TestBetaBernoulli;
  m.play();
  test_pdf(m.marginal(), N);
}
