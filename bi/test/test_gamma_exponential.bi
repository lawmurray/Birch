/*
 * Test Gamma-Exponential conjugacy.
 */
program test_gamma_exponential(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  k:Real <- simulate_uniform(0.0, 10.0);
  θ:Real <- simulate_uniform(0.0, 10.0);

  /* simulate forward */
  for n:Integer in 1..N {
    m:TestGammaExponential(k, θ);
    m.initialize();
    X1[n,1..2] <- m.forward();
  }

  /* simulate backward */
  for n:Integer in 1..N {
    m:TestGammaExponential(k, θ);
    m.initialize();
    X2[n,1..2] <- m.backward();
  }

  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestGammaExponential(k:Real, θ:Real) {
  k:Real <- k;
  θ:Real <- θ;
  λ:Random<Real>;
  x:Random<Real>;

  function initialize() {
    λ ~ Gamma(k, θ);
    x ~ Exponential(λ);
  }

  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- λ.value();
    assert x.isMissing();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    assert λ.isMissing();
    y[1] <- λ.value();
    return y;
  }
}
