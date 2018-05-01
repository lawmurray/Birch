class TestNormalInverseGammaGaussian(μ_0:Real, a2:Real, α:Real, β:Real) <
    TestConjugate {
  μ_0:Real <- μ_0;
  a2:Real <- a2;
  α:Real <- α;
  β:Real <- β;
  
  σ2:Random<Real>;
  μ:Random<Real>;
  x:Random<Real>;
  
  function initialize() {
    σ2 ~ InverseGamma(α, β);
    μ ~ Gaussian(μ_0, a2*σ2);
    x ~ Gaussian(μ, σ2);
  }
  
  function forward() -> Real[_] {
    y:Real[3];
    y[1] <- σ2.value();
    y[2] <- μ.value();
    y[3] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[3];
    y[3] <- x.value();
    y[2] <- μ.value();
    y[1] <- σ2.value();
    return y;
  }
}
