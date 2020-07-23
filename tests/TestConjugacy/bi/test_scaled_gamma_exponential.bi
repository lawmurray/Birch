/*
 * Test scaled gamma-exponential conjugacy.
 */
program test_scaled_gamma_exponential(N:Integer <- 10000) {
  m:TestScaledGammaExponential;
  test_conjugacy(m, N, 2);
}
