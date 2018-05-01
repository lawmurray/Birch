class TestGammaPoisson(k:Real, θ:Real) < TestConjugate {
  k:Real <- k;
  θ:Real <- θ; 
  λ:Random<Real>;
  x:Random<Integer>;
  
  function initialize() {
    λ ~ Gamma(k, θ);
    x ~ Poisson(λ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- λ.value();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    y[1] <- λ.value();
    return y;
  }
}
