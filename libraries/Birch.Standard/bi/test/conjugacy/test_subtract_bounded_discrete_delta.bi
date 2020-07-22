/*
 * Test analytical solution for subtraction of bounded discrete random
 * variates.
 */
program test_subtract_bounded_discrete_delta(N:Integer <- 10000) {
  m:TestSubtractBoundedDiscreteDelta;
  test_conjugacy(m, N, 2);
}

class TestSubtractBoundedDiscreteDelta < Model {
  x1:Random<Integer>;
  x2:Random<Integer>;
  s:Random<Integer>;
  
  function initialize() {
    //
  }
  
  fiber simulate() -> Event {
    x1 ~ Binomial(100, 0.75);
    x2 ~ Binomial(100, 0.25);
    s ~ Delta(x1 - x2);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    assert !x1.hasValue();
    y[1] <- x1.value();
    assert !x2.hasValue();
    y[2] <- x2.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    s.value();
    y[2] <- x2.value();
    y[1] <- x1.value();
    return y;
  }
}
