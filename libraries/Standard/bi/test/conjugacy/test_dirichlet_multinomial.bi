/*
 * Test Dirichlet-multinomial conjugacy.
 */
program test_dirichlet_multinomial(N:Integer <- 10000) {
  m:TestDirichletMultinomial;
  test_conjugacy(m, N, 10);
}
