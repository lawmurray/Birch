/**
 * Markov chain.
 */
class MarkovChainModel<StateVariate,InitialModel,TransitionModel> <
    Model<MarkovChainVariate<StateVariate>> {
  m:InitialModel;
  f:TransitionModel;

  fiber simulate(v:Variate) -> Real {
    auto xs <- v.θ.walk();
        
    /* initial state */
    x:StateVariate;
    if (xs?) {
      x <- xs!;
    }
    yield sum(m.simulate(x));
    
    /* transition */
    while (true) {
      x0:StateVariate <- x;
      if (xs?) {
        x <- xs!;
      } else {
        o:StateVariate;
        x <- o;
      }
      yield sum(f.simulate(x, x0));
    }
  }
}
