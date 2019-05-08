/*
 * Test ridge regressionm conjugacy.
 */
program test_ridge_regression(N:Integer <- 10000) {
  X:Real[N,4];
  Y1:Real[N,18];
  Y2:Real[N,18];
  
  M:Real[4,3];
  Σ:Real[4,4];
  α:Real <- simulate_uniform(2.0, 10.0);
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
    β[i] <- simulate_uniform(0.0, 10.0);
  }
  Σ <- Σ*transpose(Σ);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestRidgeRegression(M, Σ, α, β, X[i,1..columns(X)]);
    m.play();
    Y1[i,1..columns(Y1)] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestRidgeRegression(M, Σ, α, β, X[i,1..columns(X)]);
    m.play();
    Y2[i,1..columns(Y2)] <- m.backward();
  }
  
  /* test result */
  if (!pass(Y1, Y2)) {
    exit(1);
  }
}

class TestRidgeRegression(M:Real[_,_], Σ:Real[_,_], α:Real, β:Real[_],
    x:Real[_]) < Model {
  M:Real[_,_] <- M;
  Σ:Real[_,_] <- Σ;
  α:Real <- α;
  β:Real[_] <- β;
  x:Real[_] <- x;
  
  θ:Random<(Real[_,_],Real[_])>;
  y:Random<Real[_]>;
  
  fiber simulate() -> Event {
    θ ~ Ridge(M, Σ, α, β);
    y ~ Regression(θ, x);
  }
  
  function forward() -> Real[_] {
    W:Real[_,_];
    σ2:Real[_];
    z:Real[18];
    
    assert !θ.hasValue();
    (W, σ2) <- θ.value();
    for auto i in 1 .. rows(W) {
      z[(i - 1)*columns(W) + 1 .. i*columns(W)] <- W[i,1..columns(W)];
    }
    z[13..15] <- σ2;
    assert !y.hasValue();
    z[16..18] <- y.value();
    
    return z;
  }

  function backward() -> Real[_] {
    W:Real[_,_];
    σ2:Real[_];
    z:Real[18];
    
    assert !y.hasValue();
    z[16..18] <- y.value();
    assert !θ.hasValue();
    (W, σ2) <- θ.value();
    z[13..15] <- σ2;
    for auto i in 1 .. rows(W) {
      z[(i - 1)*columns(W) + 1 .. i*columns(W)] <- W[i,1..columns(W)];
    }
    
    return z;
  }
}
