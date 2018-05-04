/**
 * Test multivariate linear normal-inverse-gamma-Gaussian conjugacy.
 */
program test_multivariate_linear_normal_inverse_gamma_gaussian(N:Integer <- 10000) {
  X1:Real[N,11];
  X2:Real[N,11];
  
  A:Real[5,5];
  μ:Real[5];
  Σ:Real[5,5];
  c:Real[5];
  α:Real <- simulate_uniform(0.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
 
  for i:Integer in 1..5 {
    μ[i] <- simulate_uniform(-10.0, 10.0);
    c[i] <- simulate_uniform(-10.0, 10.0);
    for j:Integer in 1..5 {
      A[i,j] <- simulate_uniform(-10.0, 10.0);
      Σ[i,j] <- simulate_uniform(-10.0, 10.0);
    }
  }
  Σ <- Σ*trans(Σ);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestMultiariateLinearNormalInverseGammaGaussian(A, μ, Σ, c, α, β);
    m.initialize();
    X1[i,1..11] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestMultiariateLinearNormalInverseGammaGaussian(A, μ, Σ, c, α, β);
    m.initialize();
    X2[i,1..11] <- m.backward();
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
    y:Real[11];
    y[1] <- σ2.value();
    assert μ.isMissing();
    y[2..6] <- μ.value();
    assert x.isMissing();
    y[7..11] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[11];
    y[7..11] <- x.value();
    assert σ2.isMissing();
    y[1] <- σ2.value();
    assert μ.isMissing();
    y[2..6] <- μ.value();
    return y;
  }
}
