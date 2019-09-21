/*
 * Test matrix-Gaussian conjugacy.
 */
program test_matrix_gaussian_matrix_gaussian(N:Integer <- 10000) {
  auto n <- 5;
  auto p <- 2;

  X1:Real[N,2*n*p];
  X2:Real[N,2*n*p];
  
  M:Real[n,p];
  U:Real[n,n];
  V:Real[p,p];
 
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
      V[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  U <- U*transpose(U);
  V <- V*transpose(V);
 
  /* simulate forward */
  for auto i in 1..N {
    m:TestMatrixGaussianMatrixGaussian(M, U, V);
    m.play();
    X1[i,1..columns(X1)] <- m.forward();
  }

  /* simulate backward */
  for auto i in 1..N {
    m:TestMatrixGaussianMatrixGaussian(M, U, V);
    m.play();
    X2[i,1..columns(X1)] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestMatrixGaussianMatrixGaussian(M:Real[_,_], U:Real[_,_],
    V:Real[_,_]) < Model {
  auto M <- M;
  auto U <- U;
  auto V <- V;
  
  X:Random<Real[_,_]>;
  Y:Random<Real[_,_]>;
  
  fiber simulate() -> Event {
    X ~ Gaussian(M, U, V);
    Y ~ Gaussian(X, U, V);
  }
  
  function forward() -> Real[_] {
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
    return copy();
  }
  
  function copy() -> Real[_] {
    y:Real[size()];
    auto k <- 0;
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
    return rows(X)*columns(X) + rows(Y)*columns(Y);
  }
}
