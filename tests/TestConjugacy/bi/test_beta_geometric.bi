/*
 * Test beta-geometric conjugacy.
 */
program test_beta_geometric(N:Integer <- 10000) {
  m:TestBetaGeometric;
  test_conjugacy(m, N, 2);
}
