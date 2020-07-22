/*
 * Test Dirichlet-categorical conjugacy.
 */
program test_dirichlet_categorical(N:Integer <- 10000) { 
  m:TestDirichletCategorical;
  test_conjugacy(m, N, 6);
}

class TestDirichletCategorical < Model {
  ρ:Random<Real[_]>;
  x:Random<Integer>;
  α:Real[5];
  
  function initialize() {
    for n in 1..5 {
      α[n] <- simulate_uniform(1.0, 10.0);
    }
  }
  
  fiber simulate() -> Event {
    ρ ~ Dirichlet(α);
    x ~ Categorical(ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[6];
    y[1..5] <- ρ.value();
     assert !x.hasValue();
    y[6] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[6];
    y[6] <- x.value();
     assert !ρ.hasValue();
    y[1..5] <- ρ.value();
    return y;
  }
  
  function marginal() -> Distribution<Integer> {
    return x.distribution()!.graft();
  }
}
