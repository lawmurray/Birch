/*
 * Test multivariate linear normal-inverse-gamma-Gaussian conjugacy.
 */
program test_multivariate_linear_normal_inverse_gamma_gaussian(N:Integer <- 10000) {
  X1:Real[N,16];
  X2:Real[N,16];
  
  A:Real[5,10];
  μ:Real[10];
  Σ:Real[10,10];
  c:Real[5];
  α:Real <- simulate_uniform(1.0, 2.0);
  β:Real <- simulate_uniform(0.0, 2.0);
 
  for i:Integer in 1..10 {
    μ[i] <- simulate_uniform(-10.0, 10.0);
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
  Σ <- Σ*trans(Σ);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestMultiariateLinearNormalInverseGammaGaussian(A, μ, Σ, c, α, β);
    m.initialize();
    X1[i,1..16] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestMultiariateLinearNormalInverseGammaGaussian(A, μ, Σ, c, α, β);
    m.initialize();
    X2[i,1..16] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestMultiariateLinearNormalInverseGammaGaussian(A:Real[_,_],
    μ_0:Real[_], Σ:Real[_,_], c:Real[_], α:Real, β:Real) {
  A:Real[_,_] <- A;
  μ_0:Real[_] <- μ_0;
  Σ:Real[_,_] <- Σ;
  c:Real[_] <- c;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  μ:Random<Real[_]>;
  x:Random<Real[_]>;
  
  function initialize() {
    σ2 ~ InverseGamma(α, β);
    μ ~ Gaussian(μ_0, Σ*σ2);
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
}
