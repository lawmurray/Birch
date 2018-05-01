/**
 * Test Dirichlet-multinomial conjugacy.
 */
program test_dirichlet_multinomial(N:Integer <- 1000) {
  X1:Real[N,10];
  X2:Real[N,10];
  n:Integer <- simulate_uniform_int(1, 100);
  α:Real[5];
  for i:Integer in 1..5 {
    α[i] <- simulate_uniform(0.0, 10.0);
  }
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestDirichletMultinomial(n, α);
    m.initialize();
    X1[i,1..10] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestDirichletMultinomial(n, α);
    m.initialize();
    X2[i,1..10] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}
