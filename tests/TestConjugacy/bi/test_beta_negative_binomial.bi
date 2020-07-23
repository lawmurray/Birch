/*
 * Test beta-negative-binomial conjugacy.
 */
program test_beta_negative_binomial(N:Integer <- 10000) {
  m:TestBetaNegativeBinomial;
  test_conjugacy(m, N, 2);
}
