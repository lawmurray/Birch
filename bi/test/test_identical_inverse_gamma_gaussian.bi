/*
 * Test multivariate inverse-gamma-Gaussian conjugacy.
 */
program test_identical_inverse_gamma_gaussian(N:Integer <- 10000) {
  X1:Real[N,6];
  X2:Real[N,6];
  μ:Real[5];
  α:Real <- simulate_uniform(2.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
 
  for i:Integer in 1..5 {
    μ[i] <- simulate_uniform(-10.0, 10.0);
  }
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestMultivariateInverseGammaGaussian(μ, α, β);
    m.play();
    X1[i,1..6] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestMultivariateInverseGammaGaussian(μ, α, β);
    m.play();
    X2[i,1..6] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestMultivariateInverseGammaGaussian(μ:Real[_], α:Real, β:Real) < Model {
  μ:Real[_] <- μ;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  x:Random<Real[_]>;
  
  fiber simulate() -> Event {
    σ2 ~ InverseGamma(α, β);
    x ~ Gaussian(μ, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[6];
    y[1] <- σ2.value();
    assert !x.hasValue();
    y[2..6] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[6];
    y[2..6] <- x.value();
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    return y;
  }
}
