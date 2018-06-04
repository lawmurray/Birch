/*
 * Test analytical solution for subtraction of bounded discrete random
 * variates.
 */
program test_subtract_bounded_discrete_delta(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestSubtractBoundedDiscreteDelta;
    m.initialize(nil);
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestSubtractBoundedDiscreteDelta;
    m.initialize(Integer(X1[i,1] - X1[i,2]));
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestSubtractBoundedDiscreteDelta {
  x1:Random<Integer>;
  x2:Random<Integer>;
  s:Random<Integer>;
  
  function initialize(y:Integer?) {
    s <- y;
    x1 ~ Binomial(100, 0.75);
    x2 ~ Binomial(100, 0.25);
    s ~ Delta(x1 - x2);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    assert x1.isMissing();
    y[1] <- x1.value();
    assert x2.isMissing();
    y[2] <- x2.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    assert x2.isMissing();
    y[2] <- x2.value();
    assert x1.isMissing();
    y[1] <- x1.value();
    return y;
  }
}
