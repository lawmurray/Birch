class TestBetaBinomial(n:Integer, α:Real, β:Real) < TestConjugate {
  n:Integer <- n;
  α:Real <- α;
  β:Real <- β;
  ρ:Random<Real>;
  x:Random<Integer>;
  
  function initialize() {
    ρ ~ Beta(α, β);
    x ~ Binomial(n, ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];    
    y[1] <- ρ.value();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];    
    y[2] <- x.value();
    y[1] <- ρ.value();
    return y;
  }
}
