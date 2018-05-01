/**
 * Test Dirichlet-categorical conjugacy.
 */
program test_dirichlet_categorical(N:Integer <- 1000) {
  X1:Real[N,6];
  X2:Real[N,6];
  α:Real[5];
  for n:Integer in 1..5 {
    α[n] <- simulate_uniform(0.0, 10.0);
  }
 
  /* simulate forward */
  for n:Integer in 1..N {
    m:TestDirichletCategorical(α);
    m.initialize();
    X1[n,1..6] <- m.forward();
  }

  /* simulate backward */
  for n:Integer in 1..N {
    m:TestDirichletCategorical(α);
    m.initialize();
    X2[n,1..6] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}
