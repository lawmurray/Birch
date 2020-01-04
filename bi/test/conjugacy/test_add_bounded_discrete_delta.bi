/*
 * Test analytical solution for addition of bounded discrete random variates.
 */
program test_add_bounded_discrete_delta(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
 
  /* simulate forward */
  for n in 1..N {
    m:TestAddBoundedDiscreteDelta;
    delay.handle(m.simulate());
    X1[n,1..2] <- m.forward();
  }

  /* simulate backward */
  for n in 1..N {
    m:TestAddBoundedDiscreteDelta;
    delay.handle(m.simulate());
    X2[n,1..2] <- m.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestAddBoundedDiscreteDelta < Model {
  x1:Random<Integer>;
  x2:Random<Integer>;
  s:Random<Integer>;
  
  fiber simulate() -> Event {
    x1 ~ Uniform(1, 6);
    x2 ~ Uniform(1, 6);
    s ~ Delta(x1 + x2);
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
