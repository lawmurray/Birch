/*
 * Test Dirichlet-categorical pmf.
 */
program test_pdf_dirichlet_categorical(D:Integer <- 10, N:Integer <- 10000) {
  m:TestDirichletCategorical;
  m.play();
  test_pdf(m.marginal(), N);
}
