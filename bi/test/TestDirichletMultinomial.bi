class TestDirichletMultinomial(n:Integer, α:Real[_]) < TestConjugate {
  n:Integer <- n;
  α:Real[_] <- α; 
  ρ:Random<Real[_]>;
  x:Random<Integer[_]>;
  
  function initialize() {
    ρ ~ Dirichlet(α);
    x ~ Multinomial(n, ρ);
  }
  
  function forward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[2*D];
    y[1..D] <- ρ.value();
    y[D+1..2*D] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[2*D];
    y[D+1..2*D] <- x.value();
    y[1..D] <- ρ.value();
    return y;
  }
}
