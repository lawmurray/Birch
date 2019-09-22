/*
 * Test matrix normal-inverse-Wishart-Gaussian conjugacy.
 */
program test_matrix_normal_inverse_wishart_matrix_gaussian(
    N:Integer <- 10000) {
  auto n <- 5;
  auto p <- 2;

  X1:Real[N,p*p + 2*n*p];
  X2:Real[N,p*p + 2*n*p];
  
  M:Real[n,p];
  U:Real[n,n];
  k:Real <- simulate_uniform(p - 1.0, p + 9.0);
  Ψ:Real[p,p];
 
  for auto i in 1..n {
    for auto j in 1..n {
      U[i,j] <- simulate_uniform(-2.0, 2.0);
    }
    for auto j in 1..p {
      M[i,j] <- simulate_uniform(-10.0, 10.0);
    }
  }
  for auto i in 1..p {
    for auto j in 1..p {
      Ψ[i,j] <- simulate_uniform(-10.0, 10.0);
    }
  }
  U <- U*transpose(U);
  Ψ <- Ψ*transpose(Ψ);
 
  /* simulate forward */
  for auto i in 1..N {
    m:TestMatrixNormalInverseWishartMatrixGaussian(M, U, k, Ψ);
    m.play();
    X1[i,1..columns(X1)] <- m.forward();
  }

  /* simulate backward */
  for auto i in 1..N {
    m:TestMatrixNormalInverseWishartMatrixGaussian(M, U, k, Ψ);
    m.play();
    X2[i,1..columns(X1)] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestMatrixNormalInverseWishartMatrixGaussian(M:Real[_,_],
    U:Real[_,_], k:Real, Ψ:Real[_,_]) < Model {
  auto M <- M;
  auto U <- U;
  auto k <- k;
  auto Ψ <- Ψ;
  
  V:Random<Real[_,_]>;
  X:Random<Real[_,_]>;
  Y:Random<Real[_,_]>;
  
  fiber simulate() -> Event {
    V ~ InverseWishart(Ψ, k);
    X ~ Gaussian(M, U, V);
    Y ~ Gaussian(X, V);
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
  
  function copy() -> Real[_] {
    y:Real[size()];
    auto k <- 0;
    for auto i in 1..rows(V) {
      y[k + 1 .. k + columns(V)] <- V.value()[i,1..columns(X)];
      k <- k + columns(V);
    }
    for auto i in 1..rows(X) {
      y[k + 1 .. k + columns(X)] <- X.value()[i,1..columns(X)];
      k <- k + columns(X);
    }
    for auto i in 1..rows(Y) {
      y[k + 1 .. k + columns(Y)] <- Y.value()[i,1..columns(Y)];
      k <- k + columns(Y);
    }    
    return y;
  }
  
  function size() -> Integer {
    return rows(V)*columns(V) + rows(X)*columns(X) + rows(Y)*columns(Y);
  }
}
