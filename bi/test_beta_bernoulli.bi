/**
 * Test beta-Bernoulli conjugacy.
 */
program test_beta_bernoulli(N:Integer <- 1000) {
  X1:Real[N,2];
  X2:Real[N,2];
  α:Real <- simulate_uniform(0.0, 100.0);
  β:Real <- simulate_uniform(0.0, 100.0);
 
  /* simulate forward */
  for n:Integer in 1..N {
    m:TestBetaBernoulli(α, β);
    m.initialize();
    X1[n,1..2] <- m.forward();
  }

  /* simulate backward */
  for n:Integer in 1..N {
    m:TestBetaBernoulli(α, β);
    m.initialize();
    X2[n,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}
