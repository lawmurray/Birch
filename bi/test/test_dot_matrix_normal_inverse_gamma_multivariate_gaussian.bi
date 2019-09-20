/*
 * Test ridge regressionm conjugacy.
 */
program test_dot_matrix_normal_inverse_gamma_multivariate_gaussian(N:Integer <- 10000) {
  X:Real[N,4];
  Y1:Real[N,18];
  Y2:Real[N,18];
  
  M:Real[4,3];
  Σ:Real[4,4];
  α:Real[3];
  β:Real[3];
 
  for i:Integer in 1..rows(X) {
    for j:Integer in 1..columns(X) {
      X[i,j] <- simulate_gaussian(0.0, 25.0);
    }
  }
  for i:Integer in 1..rows(M) {
    for j:Integer in 1..columns(M) {
      M[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  for i:Integer in 1..rows(Σ) {
    for j:Integer in 1..columns(Σ) {
      Σ[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  for i:Integer in 1..length(β) {
    α[i] <- simulate_uniform(2.0, 10.0);
    β[i] <- simulate_uniform(0.0, 10.0);
  }
  Σ <- Σ*transpose(Σ);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestDotMatrixNormalInverseGammaMultivariateGaussian(M, Σ, α, β, X[i,1..columns(X)]);
    m.play();
    Y1[i,1..columns(Y1)] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestDotMatrixNormalInverseGammaMultivariateGaussian(M, Σ, α, β, X[i,1..columns(X)]);
    m.play();
    Y2[i,1..columns(Y2)] <- m.backward();
  }
  
  /* test result */
  if (!pass(Y1, Y2)) {
    exit(1);
  }
}

class TestDotMatrixNormalInverseGammaMultivariateGaussian(M:Real[_,_], Σ:Real[_,_], α:Real[_], β:Real[_],
    x:Real[_]) < Model {
  M:Real[_,_] <- M;
  Σ:Real[_,_] <- Σ;
  α:Real[_] <- α;
  β:Real[_] <- β;
  x:Real[_] <- x;
  
  σ2:Random<Real[_]>;
  W:Random<Real[_,_]>;
  y:Random<Real[_]>;
  
  fiber simulate() -> Event {
    σ2 ~ InverseGamma(α, β);
    W ~ Gaussian(M, Σ, σ2);
    y ~ Gaussian(W*x, σ2);
  }
  
  function forward() -> Real[_] {
    z:Real[18];
    z[1..12] <- vec(W.value());
    assert !σ2.hasValue();
    z[13..15] <- σ2.value();
    assert !y.hasValue();
    z[16..18] <- y.value();
    
    return z;
  }

  function backward() -> Real[_] {
    z:Real[18];
    
    assert !y.hasValue();
    z[16..18] <- y.value();
    assert !σ2.hasValue();
    z[13..15] <- σ2.value();
    assert !W.hasValue();
    z[1..12] <- vec(W.value());
    
    return z;
  }
}
