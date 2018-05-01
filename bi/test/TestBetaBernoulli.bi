class TestBetaBernoulli(α:Real, β:Real) < TestConjugate {
  α:Real <- α;
  β:Real <- β;
  ρ:Random<Real>;
  x:Random<Boolean>;
  
  function initialize() {
    ρ ~ Beta(α, β);
    x ~ Bernoulli(ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    
    y[1] <- ρ.value();
    if (x.value()) {
      y[2] <- 1.0;
    } else {
      y[2] <- 0.0;
    }
    
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    
    if (x.value()) {
      y[2] <- 1.0;
    } else {
      y[2] <- 0.0;
    }
    y[1] <- ρ.value();
    
    return y;
  }
}
