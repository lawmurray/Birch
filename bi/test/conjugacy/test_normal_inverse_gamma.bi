/*
 * Test normal-inverse-gamma conjugacy.
 */
program test_normal_inverse_gamma(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  m:TestNormalInverseGamma;
  m.play();
   
  /* simulate forward */
  for auto n in 1..N {
    auto m' <- clone<TestNormalInverseGamma>(m);
    X1[n,1..2] <- m'.forward();
  }

  /* simulate backward */
  for auto n in 1..N {
    auto m' <- clone<TestNormalInverseGamma>(m);
    X2[n,1..2] <- m'.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestNormalInverseGamma < Model {  
  σ2:Random<Real>;
  x:Random<Real>;
  
  fiber simulate() -> Event {
    auto μ <- simulate_uniform(-10.0, 10.0);
    auto a2 <- simulate_uniform(0.1, 2.0);
    auto α <- simulate_uniform(2.0, 10.0);
    auto β <- simulate_uniform(0.1, 10.0);

    σ2 ~ InverseGamma(α, β);
    x ~ Gaussian(μ, a2, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    assert !x.hasValue();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    assert !x.hasValue();
    y[2] <- x.value();
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    return y;
  }
  
  function marginal() -> Distribution<Real> {
    return x.distribution();
  }
}
