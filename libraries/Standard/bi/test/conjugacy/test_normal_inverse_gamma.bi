/*
 * Test normal-inverse-gamma conjugacy.
 */
program test_normal_inverse_gamma(N:Integer <- 10000) {
  m:TestNormalInverseGamma;
  test_conjugacy(m, N, 2);
}
