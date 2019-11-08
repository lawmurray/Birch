/*
 * Test beta-negative-binomial pmf.
 */
program test_pdf_beta_negative_binomial(D:Integer <- 10, N:Integer <- 10000) {
  m:TestBetaNegativeBinomial;
  m.play();
  test_pdf(m.marginal(), N);
}
