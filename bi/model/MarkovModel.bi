/**
 * A Markov model, structured as a sequence of states, where each state is
 * conditionally independent of the state history, given the previous
 * state.
 */
class MarkovModel<StateType <= State, ParameterType <= Model> < Model {
  /**
   * Parameter.
   */
  θ:ParameterType;

  /**
   * State history.
   */
  history:List<StateType>;
  
  /**
   * State future.
   */
  future:List<StateType>;

  fiber simulate() -> Real! {
    f:Real!;
    w:Real <- 0.0;
    
    /* parameter model */
    f <- θ.simulate();
    while (f?) {
      w <- w + f!;
    }

    /* initial state */
    x:StateType;
    f <- x.simulate(θ);
    while (f?) {
      w <- w + f!;
    }
    history.pushBack(x);
    yield w;
    w <- 0.0;
    
    /* transition */
    while (true) {
      /* use a preloaded future state if given, otherwise create a new
       * state */
      if (future.empty()) {
        x1:StateType;
        x <- x1;
      } else {
        x <- future.front();
        future.popFront();
      }

      /* propagate and weight */
      w <- 0.0;
      f <- x.simulate(history.back(), θ);
      while (f?) {
        w <- w + f!;
      }
      history.pushBack(x);      
      yield w;
      w <- 0.0;
    }
  }
  
  function input(reader:Reader') {
    f:Reader'! <- reader.getArray();
    while (f?) {
      x:StateType;
      x.input(f!);
      future.pushBack(x);
    }
  }
  
  function output(writer:Writer) {
    writer.setArray();
    f:StateType! <- history.walk();
    while (f?) {
      f!.output(writer.push());
    }
  }
}
