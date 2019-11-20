/*
 * Test matrix linear normal-inverse-Wishart-Gaussian conjugacy.
 */
program test_negative_linear_matrix_normal_inverse_wishart_matrix_gaussian(
    N:Integer <- 10000) {
  auto n <- 5;
  auto p <- 2;

  m:TestNegativeLinearMatrixNormalInverseWishartMatrixGaussian;
  delay.handle(m.simulate());

  /* simulate forward */
  X1:Real[N,p*p + 2*n*p];
  for i in 1..N {
    auto m' <- clone<TestNegativeLinearMatrixNormalInverseWishartMatrixGaussian>(m);
    X1[i,1..columns(X1)] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,p*p + 2*n*p];
  for i in 1..N {
    auto m' <- clone<TestNegativeLinearMatrixNormalInverseWishartMatrixGaussian>(m);
    X2[i,1..columns(X1)] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestNegativeLinearMatrixNormalInverseWishartMatrixGaussian < Model {
  V:Random<Real[_,_]>;
  X:Random<Real[_,_]>;
  Y:Random<Real[_,_]>;
  
  fiber simulate() -> Event {
    auto n <- 5;
    auto p <- 2;

    A:Real[n,n];
    M:Real[n,p];
    U:Real[n,n];
    C:Real[n,p];
    k:Real <- simulate_uniform(p - 1.0, p + 9.0);
    Ψ:Real[p,p];
 
    for i in 1..n {
      for j in 1..n {
        A[i,j] <- simulate_uniform(-2.0, 2.0);
        U[i,j] <- simulate_uniform(-2.0, 2.0);
      }
      for j in 1..p {
        M[i,j] <- simulate_uniform(-10.0, 10.0);
        C[i,j] <- simulate_uniform(-10.0, 10.0);
      }
    }
    for i in 1..p {
      for j in 1..p {
        Ψ[i,j] <- simulate_uniform(-10.0, 10.0);
      }
    }
    U <- U*transpose(U);
    Ψ <- Ψ*transpose(Ψ);

    V ~ InverseWishart(Ψ, k);
    X ~ Gaussian(M, U, V);
    Y ~ Gaussian(A*X - C, V);
  }
  
  function forward() -> Real[_] {
    assert !V.hasValue();
    V.value();
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
    assert !V.hasValue();
    V.value();
    return copy();
  }
  
  function marginal() -> Distribution<Real[_,_]> {
    return Y.distribution();
  }
  
  function copy() -> Real[_] {
    y:Real[size()];
    auto k <- 0;
    for i in 1..rows(V) {
      y[k + 1 .. k + columns(V)] <- V.value()[i,1..columns(V)];
      k <- k + columns(V);
    }
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
    return rows(V)*columns(V) + rows(X)*columns(X) + rows(Y)*columns(Y);
  }
}
