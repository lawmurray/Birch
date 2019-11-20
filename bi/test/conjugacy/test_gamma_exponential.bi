/*
 * Test Gamma-Exponential conjugacy.
 */
program test_gamma_exponential(N:Integer <- 10000) {
  m:TestGammaExponential;
  delay.handle(m.simulate());

  /* simulate forward */
  X1:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestGammaExponential>(m);
    X1[n,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestGammaExponential>(m);
    X2[n,1..2] <- m'.backward();
  }

  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestGammaExponential < Model {
  λ:Random<Real>;
  x:Random<Real>;

  fiber simulate() -> Event {
    auto k <- simulate_uniform(1.0, 10.0);
    auto θ <- simulate_uniform(0.0, 2.0);

    λ ~ Gamma(k, θ);
    x ~ Exponential(λ);
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
  
  function marginal() -> Distribution<Real> {
    return x.distribution();
  }
}
