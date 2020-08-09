/*
 * Test Gamma-Exponential conjugacy.
 */
program test_gamma_exponential(N:Integer <- 10000) {
  m:TestGammaExponential;
  test_conjugacy(m, N, 2);
}
