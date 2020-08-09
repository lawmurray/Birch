class TestBetaNegativeBinomial < Model {
  k:Integer;
  α:Real;
  β:Real;
  ρ:Random<Real>;
  x:Random<Integer>;

  function initialize() {
    //
  }

  fiber simulate() -> Event {
    k <- simulate_uniform_int(1, 100);
    α <- simulate_uniform(1.0, 100.0);
    β <- simulate_uniform(1.0, 100.0);

    ρ ~ Beta(α, β);
    x ~ NegativeBinomial(k, ρ);
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
