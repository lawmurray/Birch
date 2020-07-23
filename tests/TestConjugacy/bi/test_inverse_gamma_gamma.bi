/*
 * Test inverse-gamma-gamma conjugacy.
 */
program test_inverse_gamma_gamma(N:Integer <- 10000) {
  m:TestInverseGammaGamma;
  test_conjugacy(m, N, 2);
}
