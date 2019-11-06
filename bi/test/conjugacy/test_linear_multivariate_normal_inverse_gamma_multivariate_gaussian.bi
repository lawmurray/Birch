/*
 * Test multivariate linear normal-inverse-gamma-Gaussian conjugacy.
 */
program test_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(N:Integer <- 10000) {
  m:TestLinearMultivariateNormalInverseGammaMultivariateGaussian;
  m.play();
  
  /* simulate forward */
  X1:Real[N,16];
  for auto n in 1..N {
    auto m' <- clone<TestLinearMultivariateNormalInverseGammaMultivariateGaussian>(m);
    X1[n,1..16] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,16];
  for auto n in 1..N {
    auto m' <- clone<TestLinearMultivariateNormalInverseGammaMultivariateGaussian>(m);
    X2[n,1..16] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestLinearMultivariateNormalInverseGammaMultivariateGaussian < Model {
  σ2:Random<Real>;
  μ:Random<Real[_]>;
  x:Random<Real[_]>;
  
  fiber simulate() -> Event {
    A:Real[5,10];
    μ_0:Real[10];
    Σ:Real[10,10];
    c:Real[5];
    α:Real <- simulate_uniform(2.0, 10.0);
    β:Real <- simulate_uniform(0.0, 10.0);
 
    for i:Integer in 1..10 {
      μ_0[i] <- simulate_uniform(-10.0, 10.0);
      for j:Integer in 1..10 {
        Σ[i,j] <- simulate_uniform(-2.0, 2.0);
      }
    }
    for i:Integer in 1..5 {
      c[i] <- simulate_uniform(-10.0, 10.0);
      for j:Integer in 1..10 {
        A[i,j] <- simulate_uniform(-2.0, 2.0);
      }
    }
    Σ <- Σ*transpose(Σ);

    σ2 ~ InverseGamma(α, β);
    μ ~ Gaussian(μ_0, Σ, σ2);
    x ~ Gaussian(A*μ + c, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[16];
    y[1] <- σ2.value();
    assert !μ.hasValue();
    y[2..11] <- μ.value();
    assert !x.hasValue();
    y[12..16] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[16];
    y[12..16] <- x.value();
    assert !μ.hasValue();
    y[2..11] <- μ.value();
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    return y;
  }
  
  function marginal() -> Distribution<Real[_]> {
    return x.distribution();
  }
}
