class TestBetaGeometric < Model {
  ρ:Random<Real>;
  x:Random<Integer>;
  α:Real;
  β:Real;

  function initialize() {
    α <- simulate_uniform(1.0, 100.0);
    β <- simulate_uniform(1.0, 100.0);
  }

  fiber simulate() -> Event {
    ρ ~ Beta(α, β);
    x ~ Geometric(ρ);
  }

  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- ρ.value();
    assert !x.hasValue();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    assert !ρ.hasValue();
    y[1] <- ρ.value();
    return y;
  }
}
