class TestBetaBinomial < Model {
  ρ:Random<Real>;
  x:Random<Integer>;
  n:Integer;
  α:Real;
  β:Real;

  function initialize() {
    n <- simulate_uniform_int(1, 100);
    α <- simulate_uniform(1.0, 10.0);
    β <- simulate_uniform(1.0, 10.0);
  }

  fiber simulate() -> Event {
    ρ ~ Beta(α, β);
    x ~ Binomial(n, ρ);
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

  function marginal() -> Distribution<Integer> {
    return x.distribution()!.graft();
  }
}
