class MarkovModel<StateType <= State> < Model {
  /**
   * State history.
   */
  path:List<StateType>;

  fiber simulate() -> Real! {
    /* initial state */
    x:StateType;
    w:Real <- 0.0;
    f:Real! <- x.simulate();
    while (f?) {
      w <- w + f!;
    }
    path.pushBack(x);
    yield w;
    
    /* transition */
    while (true) {
      x:StateType;
      w <- 0.0;
      f <- x.simulate(path.back());
      while (f?) {
        w <- w + f!;
      }
      path.pushBack(x);
      yield w;
    }
  }
}
