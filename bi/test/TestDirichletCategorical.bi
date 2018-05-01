class TestDirichletCategorical(α:Real[_]) < TestConjugate {
  α:Real[_] <- α; 
  ρ:Random<Real[_]>;
  x:Random<Integer>;
  
  function initialize() {
    ρ ~ Dirichlet(α);
    x ~ Categorical(ρ);
  }
  
  function forward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[D + 1];
    y[1..D] <- ρ.value();
    y[D + 1] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[D + 1];
    y[D + 1] <- x.value();
    y[1..D] <- ρ.value();
    return y;
  }
}
