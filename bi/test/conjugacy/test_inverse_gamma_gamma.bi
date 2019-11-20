/*
 * Test inverse-gamma-gamma conjugacy.
 */
program test_inverse_gamma_gamma(N:Integer <- 10000) {
  m:TestInverseGammaGamma;
  delay.handle(m.simulate());
 
  /* simulate forward */
  X1:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestInverseGammaGamma>(m);
    X1[n,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestInverseGammaGamma>(m);
    X2[n,1..2] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestInverseGammaGamma < Model {
  θ:Random<Real>;
  x:Random<Real>;
  
  fiber simulate() -> Event {
    auto k <- simulate_uniform(1.0, 10.0);
    auto α <- simulate_uniform(2.0, 10.0);
    auto β <- simulate_uniform(0.0, 10.0);
  
    θ ~ InverseGamma(α, β);
    x ~ Gamma(k, θ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- θ.value();
    assert !x.hasValue();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    assert !θ.hasValue();
    y[1] <- θ.value();
    return y;
  }
  
  function marginal() -> Distribution<Real> {
    return x.distribution();
  }
}
