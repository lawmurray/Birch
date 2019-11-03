/*
 * Test normal-inverse-gamma conjugacy.
 */
program test_normal_inverse_gamma(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  
  μ:Real <- simulate_uniform(-10.0, 10.0);
  a2:Real <- simulate_uniform(0.0, 2.0);
  α:Real <- simulate_uniform(2.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestNormalInverseGamma(μ, a2, α, β);
    m.play();
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestNormalInverseGamma(μ, a2, α, β);
    m.play();
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestNormalInverseGamma(μ_0:Real, a2:Real, α:Real, β:Real) < Model {
  μ_0:Real <- μ_0;
  a2:Real <- a2;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  μ:Random<Real>;
  
  fiber simulate() -> Event {
    σ2 ~ InverseGamma(α, β);
    assert !σ2.hasValue();
    μ ~ Gaussian(μ_0, a2, σ2);
    assert !σ2.hasValue();
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    assert !μ.hasValue();
    y[2] <- μ.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    assert !μ.hasValue();
    y[2] <- μ.value();
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    return y;
  }
}
