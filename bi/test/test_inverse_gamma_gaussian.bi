/*
 * Test inverse-gamma-Gaussian conjugacy.
 */
program test_inverse_gamma_gaussian(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  μ:Real <- simulate_uniform(-10.0, 10.0);
  α:Real <- simulate_uniform(0.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestInverseGammaGaussian(μ, α, β);
    m.initialize();
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestInverseGammaGaussian(μ, α, β);
    m.initialize();
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestInverseGammaGaussian(μ:Real, α:Real, β:Real) {
  μ:Real <- μ;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  x:Random<Real>;
  
  function initialize() {
    σ2 ~ InverseGamma(α, β);
    x ~ Gaussian(μ, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- σ2.value();
    assert x.isMissing();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    assert σ2.isMissing();
    y[1] <- σ2.value();
    return y;
  }
}
