/*
 * Test multivariate dot normal-inverse-gamma-Gaussian conjugacy.
 */
program test_multivariate_dot_normal_inverse_gamma_gaussian(N:Integer <- 10000) {
  X1:Real[N,7];
  X2:Real[N,7];
  
  a:Real[5];
  μ:Real[5];
  Σ:Real[5,5];
  c:Real <- simulate_uniform(-10.0, 10.0);
  α:Real <- simulate_uniform(1.0, 2.0);
  β:Real <- simulate_uniform(0.0, 2.0);
 
  for i:Integer in 1..5 {
    μ[i] <- simulate_uniform(-10.0, 10.0);
    a[i] <- simulate_uniform(-2.0, 2.0);
    for j:Integer in 1..5 {
      Σ[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  Σ <- Σ*trans(Σ);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestMultivariateDotNormalInverseGammaGaussian(a, μ, Σ, c, α, β);
    m.initialize();
    X1[i,1..7] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestMultivariateDotNormalInverseGammaGaussian(a, μ, Σ, c, α, β);
    m.initialize();
    X2[i,1..7] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestMultivariateDotNormalInverseGammaGaussian(a:Real[_],
    μ_0:Real[_], Σ:Real[_,_], c:Real, α:Real, β:Real) {
  a:Real[_] <- a;
  μ_0:Real[_] <- μ_0;
  Σ:Real[_,_] <- Σ;
  c:Real <- c;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  μ:Random<Real[_]>;
  x:Random<Real>;
  
  function initialize() {
    σ2 ~ InverseGamma(α, β);
    μ ~ Gaussian(μ_0, Σ*σ2);
    x ~ Gaussian(dot(a, μ) + c, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[7];
    y[1] <- σ2.value();
    assert !μ.hasValue();
    y[2..6] <- μ.value();
    assert !x.hasValue();
    y[7] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[7];
    y[7] <- x.value();
    assert !σ2.hasValue();
    y[1] <- σ2.value();
    assert !μ.hasValue();
    y[2..6] <- μ.value();
    return y;
  }
}
