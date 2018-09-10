/**
 * State-space model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}x_{0:T}, \mathrm{d}\theta) = p(\mathrm{d}\theta)
 *   p(\mathrm{d}x_0 \mid \theta) \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1},
 *   \theta)$$
 *
 * <center>
 * ![Graphical model depiction of MarkovModel.](../figs/MarkovModel.svg)
 * </center>
 */
class StateSpaceModel<Parameter,Initial,Transition,Observation> <
    Model<StateSpaceVariate<Parameter.Variate,Initial.Variate,Observation.Variate>> {

  fiber simulate(v:Variate) -> Real {
    auto f <- v.x.walk();
    auto g <- v.y.walk();
    
    /* parameter */
    yield sum(p.p(v.θ));
    
    /* initial state and initial observation */
    x:State;
    y:Observation;
    if (f?) {
      x <- f!;
    }
    if (g?) {
      y <- g!;
    }
    yield sum(m.p(x, v.θ)) + sum(g.p(y, x, v.θ));
    
    /* transition and observation */
    while (true) {
      x0:State <- x;
      if (f?) {
        x <- f!;
      } else {
        o:State;
        x <- o;
      }
      if (g?) {
        y <- g!;
      } else {
        o:Observation;
        y <- o;
      }
      yield sum(f.p(x, x0, v.θ)) + sum(g.p(y, x, v.θ));
    }
  }
}
