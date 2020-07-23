class TestDirichletMultinomial < Model {
  n:Integer;
  α:Real[5];
  ρ:Random<Real[_]>;
  x:Random<Integer[_]>;

  function initialize() {
    n <- simulate_uniform_int(100, 500);
    for i in 1..5 {
      α[i] <- simulate_uniform(1.0, 10.0);
    }
  }

  fiber simulate() -> Event {
    ρ ~ Dirichlet(α);
    x ~ Multinomial(n, ρ);
  }

  function forward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[2*D];
    y[1..D] <- ρ.value();
    assert !x.hasValue();
    y[D+1..2*D] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[2*D];
    y[D+1..2*D] <- x.value();
    assert !ρ.hasValue();
    y[1..D] <- ρ.value();
    return y;
  }
}
