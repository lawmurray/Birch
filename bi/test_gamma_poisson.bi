/**
 * Test gamma-Poisson conjugacy.
 */
program test_gamma_poisson(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  k:Real <- simulate_uniform_int(1, 10);
  θ:Real <- simulate_uniform(0.0, 10.0);
 
  /* simulate forward */
  for n:Integer in 1..N {
    m:TestGammaPoisson(k, θ);
    m.initialize();
    X1[n,1..2] <- m.forward();
  }

  /* simulate backward */
  for n:Integer in 1..N {
    m:TestGammaPoisson(k, θ);
    m.initialize();
    X2[n,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}
