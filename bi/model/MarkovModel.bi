/**
 * Markov model, structured as a sequence of states, where each state is
 * conditionally independent of the state history, given the previous state.
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
    x:StateType;
    w:Real <- 0.0;
    
    /* parameter model */
    f <- θ.simulate();
    while (f?) {
      w <- w + f!;
    }

    /* initial state */
    if (!future.empty()) {
      x <- future.front();
      future.popFront();
    }
    f <- x.simulate(θ);
    while (f?) {
      w <- w + f!;
    }
    history.pushBack(x);
    yield w;
    
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
    }
  }
  
  function input(reader:Reader) {
    parameter:Reader? <- reader.getObject("parameter");
    if (parameter?) {
      θ.input(parameter!);
    }
    
    state:Reader? <- reader.getObject("state");
    if (!state?) {
      /* try root instead */
      state <- reader;
    }
    f:Reader! <- state!.getArray();
    while (f?) {
      x:StateType;
      x.input(f!);
      future.pushBack(x);
    }
  }
  
  function output(writer:Writer) {
    θ.output(writer.setObject("parameter"));
    
    state:Writer <- writer.setArray("state");
    f:StateType! <- history.walk();
    while (f?) {
      f!.output(state.push());
    }
  }
}
