/*
 * Test scaled gamma-Poisson conjugacy.
 */
program test_negative_scaled_gamma_poisson(N:Integer <- 10000) {
  m:TestNegativeScaledGammaPoisson;
  delay.handle(m.simulate());
 
  /* simulate forward */
  X1:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestNegativeScaledGammaPoisson>(m);
    X1[n,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestNegativeScaledGammaPoisson>(m);
    X2[n,1..2] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestNegativeScaledGammaPoisson < Model {
  λ:Random<Real>;
  x:Random<Integer>;
  
  fiber simulate() -> Event {
    a:Real <- simulate_uniform(1.0, 100.0);
    k:Real <- simulate_uniform_int(1, 10);
    θ:Real <- simulate_uniform(0.0, 10.0);

    λ ~ Gamma(k, θ);
    x ~ Poisson(λ/a);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- λ.value();
    assert !x.hasValue();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    assert !λ.hasValue();
    y[1] <- λ.value();
    return y;
  }
  
  function marginal() -> Distribution<Integer> {
    return x.distribution();
  }
}
