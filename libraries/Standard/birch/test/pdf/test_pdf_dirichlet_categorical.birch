/*
 * Test Dirichlet-categorical pmf.
 */
program test_pdf_dirichlet_categorical(N:Integer <- 10000) {
  m:TestDirichletCategorical;
  m.initialize();
  handle(m.simulate());
  test_pdf(m.marginal(), N);
}
