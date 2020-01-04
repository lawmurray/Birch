/*
 * Test linear inverse-gamma-Gaussian conjugacy.
 */
program test_linear_normal_inverse_gamma_gaussian(N:Integer <- 10000) {  
  m:TestLinearNormalInverseGammaGaussian;
  delay.handle(m.simulate());
  
  /* simulate forward */
  X1:Real[N,3];
  for n in 1..N {
    auto m' <- clone<TestLinearNormalInverseGammaGaussian>(m);
    X1[n,1..3] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,3];
  for n in 1..N {
    auto m' <- clone<TestLinearNormalInverseGammaGaussian>(m);
    X2[n,1..3] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestLinearNormalInverseGammaGaussian < Model {  
  σ2:Random<Real>;
  μ:Random<Real>;
  x:Random<Real>;
  
  fiber simulate() -> Event {
    auto a <- simulate_uniform(-2.0, 2.0);
    auto μ_0 <- simulate_uniform(-10.0, 10.0);
    auto a2 <- simulate_uniform(0.1, 2.0);
    auto c <- simulate_uniform(-10.0, 10.0);
    auto α <- simulate_uniform(2.0, 10.0);
    auto β <- simulate_uniform(0.1, 10.0);

    σ2 ~ InverseGamma(α, β);
    μ ~ Gaussian(μ_0, a2, σ2);
    x ~ Gaussian(a*μ + c, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[3];
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    assert !μ.hasValue();
    y[2] <- μ.value();
    assert !x.hasValue();
    y[3] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[3];
    assert !x.hasValue();
    y[3] <- x.value();
    assert !μ.hasValue();
    y[2] <- μ.value();
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    return y;
  }
  
  function marginal() -> Distribution<Real> {
    return x.distribution()!.graft();
  }
}
