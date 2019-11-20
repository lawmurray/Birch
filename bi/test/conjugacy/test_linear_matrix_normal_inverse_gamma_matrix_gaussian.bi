/*
 * Test matrix linear normal-inverse-gamma-Gaussian conjugacy.
 */
program test_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    N:Integer <- 10000) {
  auto n <- 5;
  auto p <- 2;

  m:TestLinearMatrixNormalInverseGammaMatrixGaussian;
  delay.handle(m.simulate());
   
  /* simulate forward */
  X1:Real[N,p + 2*n*p];
  for i in 1..N {
    auto m' <- clone<TestLinearMatrixNormalInverseGammaMatrixGaussian>(m);
    X1[i,1..columns(X1)] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,p + 2*n*p];
  for i in 1..N {
    auto m' <- clone<TestLinearMatrixNormalInverseGammaMatrixGaussian>(m);
    X2[i,1..columns(X1)] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestLinearMatrixNormalInverseGammaMatrixGaussian < Model {
  σ2:Random<Real[_]>;
  X:Random<Real[_,_]>;
  Y:Random<Real[_,_]>;
  
  fiber simulate() -> Event {
    auto n <- 5;
    auto p <- 2;

    A:Real[n,n];
    M:Real[n,p];
    Σ:Real[n,n];
    C:Real[n,p];
    α:Real <- simulate_uniform(2.0, 10.0);
    β:Real[p];
 
    for i in 1..n {
      for j in 1..n {
        A[i,j] <- simulate_uniform(-2.0, 2.0);
        Σ[i,j] <- simulate_uniform(-2.0, 2.0);
      }
      for j in 1..p {
        M[i,j] <- simulate_uniform(-10.0, 10.0);
        C[i,j] <- simulate_uniform(-10.0, 10.0);
      }
    }
    for i in 1..p {
      β[i] <- simulate_uniform(0.0, 10.0);
    }
    Σ <- Σ*transpose(Σ);

    σ2 ~ InverseGamma(α, β);
    X ~ Gaussian(M, Σ, σ2);
    Y ~ Gaussian(A*X + C, σ2);
  }
  
  function forward() -> Real[_] {
    assert !σ2.hasValue();
    σ2.value();
    assert !X.hasValue();
    X.value();
    assert !Y.hasValue();
    Y.value();
    return copy();
  }

  function backward() -> Real[_] {
    assert !Y.hasValue();
    Y.value();
    assert !X.hasValue();
    X.value();
    assert !σ2.hasValue();
    σ2.value();
    return copy();
  }
  
  function marginal() -> Distribution<Real[_,_]> {
    return Y.distribution();
  }
  
  function copy() -> Real[_] {
    y:Real[size()];
    y[1..length(σ2)] <- σ2;
    auto k <- length(σ2);
    for i in 1..rows(X) {
      y[k + 1 .. k + columns(X)] <- X.value()[i,1..columns(X)];
      k <- k + columns(X);
    }
    for i in 1..rows(Y) {
      y[k + 1 .. k + columns(Y)] <- Y.value()[i,1..columns(Y)];
      k <- k + columns(Y);
    }    
    return y;
  }
  
  function size() -> Integer {
    return length(σ2) + rows(X)*columns(X) + rows(Y)*columns(Y);
  }
}
