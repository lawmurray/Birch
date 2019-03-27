/*
 * Test multivariate dot Gaussian-Gaussian conjugacy.
 */
program test_multivariate_dot_gaussian_gaussian(N:Integer <- 10000) {
  X1:Real[N,6];
  X2:Real[N,6];
  
  a:Real[5];
  μ_0:Real[5];
  Σ_0:Real[5,5];
  c:Real <- simulate_uniform(-10.0, 10.0);
  σ2_1:Real <- simulate_uniform(-2.0, 2.0);

  for i:Integer in 1..5 {
    μ_0[i] <- simulate_uniform(-10.0, 10.0);
    a[i] <- simulate_uniform(-2.0, 2.0);
    for j:Integer in 1..5 {
      Σ_0[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  Σ_0 <- Σ_0*trans(Σ_0);
  σ2_1 <- σ2_1*σ2_1;
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestMultivariateDotGaussianGaussian(a, μ_0, Σ_0, c, σ2_1);
    m.play();
    X1[i,1..6] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestMultivariateDotGaussianGaussian(a, μ_0, Σ_0, c, σ2_1);
    m.play();
    X2[i,1..6] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestMultivariateDotGaussianGaussian(a:Real[_], μ_0:Real[_],
    Σ_0:Real[_,_], c:Real, σ2_1:Real) < Model {
  a:Real[_] <- a;
  μ_0:Real[_] <- μ_0;
  Σ_0:Real[_,_] <- Σ_0;
  c:Real <- c;
  σ2_1:Real <- σ2_1;
  
  μ_1:Random<Real[_]>;
  x:Random<Real>;
  
  fiber simulate() -> Event {
    μ_1 ~ Gaussian(μ_0, Σ_0);
    x ~ Gaussian(dot(a, μ_1) + c, σ2_1);
  }
  
  function forward() -> Real[_] {
    y:Real[6];
    y[1..5] <- μ_1.value();
    assert !x.hasValue();
    y[6] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[6];
    y[6] <- x.value();
    assert !μ_1.hasValue();
    y[1..5] <- μ_1.value();
    return y;
  }
}
