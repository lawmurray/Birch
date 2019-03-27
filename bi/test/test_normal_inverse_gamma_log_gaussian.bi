/*
 * Test inverse-gamma-Gaussian conjugacy.
 */
program test_normal_inverse_gamma_log_gaussian(N:Integer <- 10000) {
  X1:Real[N,3];
  X2:Real[N,3];
  
  μ:Real <- simulate_uniform(-10.0, 10.0);
  a2:Real <- simulate_uniform(0.0, 2.0);
  α:Real <- simulate_uniform(2.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
  
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestNormalInverseGammaLogGaussian(μ, a2, α, β);
    m.play();
    X1[i,1..3] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestNormalInverseGammaLogGaussian(μ, a2, α, β);
    m.play();
    X2[i,1..3] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestNormalInverseGammaLogGaussian(μ_0:Real, a2:Real, α:Real, β:Real) < Model {
  μ_0:Real <- μ_0;
  a2:Real <- a2;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  μ:Random<Real>;
  x:Random<Real>;
  
  fiber simulate() -> Event {
    σ2 ~ InverseGamma(α, β);
    μ ~ Gaussian(μ_0, a2*σ2);
    x ~ LogGaussian(μ, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[3];
    y[1] <- σ2.value();
    assert !μ.hasValue();
    y[2] <- μ.value();
    assert !x.hasValue();
    y[3] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[3];
    y[3] <- x.value();
    assert !μ.hasValue();
    y[2] <- μ.value();
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    return y;
  }
}
