/*
 * Test linear inverse-gamma-Gaussian conjugacy.
 */
program test_linear_normal_inverse_gamma_gaussian(N:Integer <- 10000) {  
  m:TestLinearNormalInverseGammaGaussian;
  test_conjugacy(m, N, 3);
}

class TestLinearNormalInverseGammaGaussian < Model {  
  σ2:Random<Real>;
  μ:Random<Real>;
  x:Random<Real>;

  a:Real;
  μ_0:Real;
  a2:Real;
  c:Real;
  α:Real;
  β:Real;
  
  function initialize() {
    a <- simulate_uniform(-2.0, 2.0);
    μ_0 <- simulate_uniform(-10.0, 10.0);
    a2 <- simulate_uniform(0.1, 2.0);
    c <- simulate_uniform(-10.0, 10.0);
    α <- simulate_uniform(2.0, 10.0);
    β <- simulate_uniform(0.1, 10.0);
  }
  
  fiber simulate() -> Event {
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
