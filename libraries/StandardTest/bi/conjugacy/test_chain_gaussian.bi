/*
 * Test a chain of conjugate Gaussians.
 */
program test_chain_gaussian(N:Integer <- 10000) {
  m:TestChainGaussian;
  test_conjugacy(m, N, 5);
}
