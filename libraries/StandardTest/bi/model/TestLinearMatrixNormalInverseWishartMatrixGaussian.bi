class TestLinearMatrixNormalInverseWishartMatrixGaussian < Model {
  V:Random<LLT>;
  X:Random<Real[_,_]>;
  Y:Random<Real[_,_]>;

  n:Integer <- 5;
  p:Integer <- 2;
  A:Real[n,n];
  M:Real[n,p];
  U:Real[n,n];
  C:Real[n,p];
  k:Real;
  Ψ:Real[p,p];

  function initialize() {
    k <- simulate_uniform(p - 1.0, p + 9.0);
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
  }

  fiber simulate() -> Event {
    V ~ InverseWishart(Ψ, k);
    X ~ Gaussian(M, U, V);
    Y ~ Gaussian(A*X + C, V);
  }

  function forward() -> Real[_] {
    assert !V.hasValue();
    V.value();
    assert !X.hasValue();
    X.value();
    assert !Y.hasValue();
    Y.value();
    return vectorize();
  }

  function backward() -> Real[_] {
    assert !Y.hasValue();
    Y.value();
    assert !X.hasValue();
    X.value();
    assert !V.hasValue();
    V.value();
    return vectorize();
  }

  function marginal() -> Distribution<Real[_,_]> {
    return Y.distribution()!;
  }

  function vectorize() -> Real[_] {
    y:Real[size()];
    auto k <- 0;
    for i in 1..rows(V) {
      y[k + 1 .. k + columns(V)] <- canonical(V.value())[i,1..columns(V)];
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
    return p*p + 2*n*p;
  }
}
