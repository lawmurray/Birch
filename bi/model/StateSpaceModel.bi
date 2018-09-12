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
  p:Parameter;
  m:Initial;
  f:Transition;
  g:Observation;

  fiber simulate(v:Variate) -> Real {
    auto xs <- v.x.walk();
    auto ys <- v.y.walk();
    
    /* parameter */
    yield sum(p.simulate(v.θ));
    
    /* initial state and initial observation */
    x:Initial.Variate;
    y:Observation.Variate;
    if (xs?) {
      x <- xs!;
    }
    if (ys?) {
      y <- ys!;
    }
    yield sum(m.simulate(x, v.θ)) + sum(g.simulate(y, x, v.θ));
    
    /* transition and observation */
    while (true) {
      x0:Initial.Variate <- x;
      if (xs?) {
        x <- xs!;
      } else {
        o:Initial.Variate;
        x <- o;
      }
      if (ys?) {
        y <- ys!;
      } else {
        o:Observation.Variate;
        y <- o;
      }
      yield sum(f.simulate(x, x0, v.θ)) + sum(g.simulate(y, x, v.θ));
    }
  }
}
