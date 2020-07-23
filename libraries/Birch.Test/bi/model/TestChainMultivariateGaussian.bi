class TestChainMultivariateGaussian < Model {
  x:Random<Real[_]>[5];
  μ:Real[3];
  Σ:Real[3,3];

  function initialize() {
    for i in 1..3 {
      μ[i] <- simulate_uniform(-10.0, 10.0);
      for j in 1..3 {
        Σ[i,j] <- simulate_uniform(-2.0, 2.0);
      }
    }
    Σ <- Σ*transpose(Σ);
  }

  fiber simulate() -> Event {
    x[1] ~ Gaussian(μ, Σ);
    x[2] ~ Gaussian(x[1], Σ);
    x[3] ~ Gaussian(x[2], Σ);
    x[4] ~ Gaussian(x[3], Σ);
    x[5] ~ Gaussian(x[4], Σ);
  }

  function forward() -> Real[_] {
    y:Real[15];
    for i in 1..5 {
      assert !x[i].hasValue();
      y[(i-1)*3+1..i*3] <- x[i].value();
    }
    return y;
  }

  function backward() -> Real[_] {
    y:Real[15];
    for i in 0..4 {
      assert !x[5-i].hasValue();
      y[(4-i)*3+1..(5-i)*3] <- x[5-i].value();
    }
    return y;
  }
}
