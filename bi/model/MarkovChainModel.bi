/**
 * Markov chain.
 */
class MarkovChainModel<State> < Model {
  /**
   * Variate.
   */
  v:MarkovChainVariate<State>;

  /**
   * Initial model.
   */
  m:@(State) -> Real! <- m;
  
  /**
   * Transition model.
   */
  f:@(State, State) -> Real! <- f;

  fiber simulate() -> Real {
    auto xs <- v.x.walk();
        
    /* initial state */
    x':State;
    if (xs?) {
      x' <- xs!;
    }
    yield sum(m(x'));
    
    /* transition */
    while (true) {
      x:State <- x';
      if (xs?) {
        x' <- xs!;
      } else {
        o:State;
        x' <- o;
      }
      yield sum(f(x', x));
    }
  }
}
