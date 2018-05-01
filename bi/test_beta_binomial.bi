/**
 * Test beta-binomial conjugacy.
 */
program test_beta_binomial(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  n:Integer <- simulate_uniform_int(1, 100);
  α:Real <- simulate_uniform(0.0, 100.0);
  β:Real <- simulate_uniform(0.0, 100.0);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestBetaBinomial(n, α, β);
    m.initialize();
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestBetaBinomial(n, α, β);
    m.initialize();
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}
