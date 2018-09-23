/**
 * Markov chain.
 *
 * - SpecificVariate: Specialization of MarkovChainVariate.
 */
class MarkovChainModel<SpecificVariate>(
    m:@(SpecificVariate.State) -> Real!,
    f:@(SpecificVariate.State, SpecificVariate.State) -> Real!) <
    ModelFor<SpecificVariate> {
  /**
   * Initial model.
   */
  auto m <- m;
  
  /**
   * Transition model.
   */
  auto f <- f;

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
}
