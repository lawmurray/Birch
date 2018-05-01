class TestGaussianGaussian(μ_0:Real, σ2_0:Real, σ2_1:Real) < TestConjugate {
  μ_0:Real <- μ_0;
  σ2_0:Real <- σ2_0;
  σ2_1:Real <- σ2_1;
  
  μ_1:Random<Real>;
  x:Random<Real>;
  
  function initialize() {
    μ_1 ~ Gaussian(μ_0, σ2_0);
    x ~ Gaussian(μ_1, σ2_1);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- μ_1.value();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    y[1] <- μ_1.value();
    return y;
  }
}
