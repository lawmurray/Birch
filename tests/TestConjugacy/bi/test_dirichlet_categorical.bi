/*
 * Test Dirichlet-categorical conjugacy.
 */
program test_dirichlet_categorical(N:Integer <- 10000) {
  m:TestDirichletCategorical;
  test_conjugacy(m, N, 6);
}
