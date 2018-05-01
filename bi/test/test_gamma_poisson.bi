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

class TestGammaPoisson(k:Real, θ:Real) {
  k:Real <- k;
  θ:Real <- θ; 
  λ:Random<Real>;
  x:Random<Integer>;
  
  function initialize() {
    λ ~ Gamma(k, θ);
    x ~ Poisson(λ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- λ.value();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    y[1] <- λ.value();
    return y;
  }
}
