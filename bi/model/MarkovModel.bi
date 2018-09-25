/**
 * Markov model.
 *
 * - SpecificVariate: Specialization of MarkovVariate.
 */
class MarkovModel<SpecificVariate> < ModelFor<SpecificVariate> {
  fiber simulate(v:SpecificVariate) -> Real {
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

  /**
   * Initial model.
   */
  fiber m(x':SpecificVariate.State, θ:SpecificVariate.Parameter) -> Real {
    //
  }
  
  /**
   * Transition model.
   */
  fiber f(x':SpecificVariate.State, x:SpecificVariate.State,
      θ:SpecificVariate.Parameter) -> Real {
    //
  }
}
