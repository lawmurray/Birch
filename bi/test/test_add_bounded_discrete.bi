/*
 * Test analytical solution for addition of bounded discrete random variates.
 */
program test_add_bounded_discrete(N:Integer <- 10000) {
  X1:Real[N,3];
  X2:Real[N,3];
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestAddBoundedDiscrete;
    m.initialize();
    X1[i,1..3] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestAddBoundedDiscrete;
    m.initialize();
    X2[i,1..3] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestAddBoundedDiscrete {
  x1:Random<Integer>;
  x2:Random<Integer>;
  s:Random<Integer>;
  
  function initialize() {
    x1 ~ Uniform(1, 6);
    x2 ~ Uniform(1, 6);
    s ~ Delta(x1 + x2);
  }
  
  function forward() -> Real[_] {
    y:Real[3];
    y[1] <- x1.value();
    assert x2.isMissing();
    y[2] <- x2.value();
    assert s.isMissing();
    y[3] <- s.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[3];
    y[3] <- s.value();
    y[2] <- x2.value();
    y[1] <- x1.value();
    return y;
  }
}
