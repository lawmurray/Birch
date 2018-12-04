/*
 * Test inverse-gamma-gamma conjugacy.
 */
program test_inverse_gamma_gamma(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  k:Real <- simulate_uniform(1.0, 10.0);
  α:Real <- simulate_uniform(0.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestInverseGammaGamma(k, α, β);
    m.initialize();
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestInverseGammaGamma(k, α, β);
    m.initialize();
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestInverseGammaGamma(k:Real, α:Real, β:Real) {
  k:Real <- k;
  α:Real <- α;
  β:Real <- β;
  
  θ:Random<Real>;
  x:Random<Real>;
  
  function initialize() {
    θ ~ InverseGamma(α, β);
    x ~ Gamma(k, θ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- θ.value();
    assert x.isMissing();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    assert θ.isMissing();
    y[1] <- θ.value();
    return y;
  }
}
