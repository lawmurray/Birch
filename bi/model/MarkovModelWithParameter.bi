class MarkovModelWithParameter<StateType <= StateWithParameter,
    ParameterType <= Parameter> {
  /**
   * Parameter.
   */
  θ:ParameterType;

  /**
   * State history.
   */
  path:List<StateType>;

  fiber simulate() -> Real! {
    /* parameter model */
    w:Real <- 0.0;
    f:Real! <- θ.simulate();
    while (f?) {
      w <- w + f!;
    }
    yield w;
    
    /* initial state */
    x:StateType;
    w <- 0.0;
    f <- x.simulate(θ);
    while (f?) {
      w <- w + f!;
    }
    path.pushBack(x);
    yield w;
    
    /* transition */
    while (true) {
      x:StateType;
      w <- 0.0;
      f <- x.simulate(path.back(), θ);
      while (f?) {
        w <- w + f!;
      }
      path.pushBack(x);
      yield w;
    }
  }
}
