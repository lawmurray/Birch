class TestInverseGammaGaussian(μ:Real, α:Real, β:Real) < TestConjugate {
  μ:Real <- μ;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  x:Random<Real>;
  
  function initialize() {
    σ2 ~ InverseGamma(α, β);
    x ~ Gaussian(μ, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- σ2.value();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    y[1] <- σ2.value();
    return y;
  }
}
