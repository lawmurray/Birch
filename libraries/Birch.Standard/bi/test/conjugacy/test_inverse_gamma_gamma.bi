/*
 * Test inverse-gamma-gamma conjugacy.
 */
program test_inverse_gamma_gamma(N:Integer <- 10000) {
  m:TestInverseGammaGamma;
  test_conjugacy(m, N, 2);
}

class TestInverseGammaGamma < Model {
  θ:Random<Real>;
  x:Random<Real>;
  k:Real;
  α:Real;
  β:Real;
  
  function initialize() {
    k <- simulate_uniform(1.0, 10.0);
    α <- simulate_uniform(2.0, 10.0);
    β <- simulate_uniform(0.0, 10.0);  
  }
  
  fiber simulate() -> Event {
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
    return x.distribution()!.graft();
  }
}
