/*
 * Test analytical solution for addition of bounded discrete random variates.
 */
program test_add_bounded_discrete_delta(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestAddBoundedDiscreteDelta;
    m.initialize(nil);
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestAddBoundedDiscreteDelta;
    m.initialize(Integer(X1[i,1] + X1[i,2]));
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestAddBoundedDiscreteDelta {
  x1:Random<Integer>;
  x2:Random<Integer>;
  s:Random<Integer>;
  
  function initialize(y:Integer?) {
    s <- y;
    x1 ~ Uniform(1, 6);
    x2 ~ Uniform(1, 6);
    s ~ Delta(x1 + x2);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- x1.value();
    assert !x2.hasValue();
    y[2] <- x2.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x2.value();
    y[1] <- x1.value();
    return y;
  }
}
