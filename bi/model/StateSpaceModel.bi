/**
 * State-space model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{0:T}, \mathrm{d}y_{0:T}) =
 *   p(\mathrm{d}\theta) p(\mathrm{d}x_0 \mid \theta)  p(\mathrm{d}y_0
 *   \mid x_0, \theta) \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1},
 *   \theta) p(\mathrm{d}y_t \mid x_t, \theta)$$
 */
class StateSpaceModel<ParameterVariate,StateVariate,ObservationVariate,
    ParameterModel,InitialModel,TransitionModel,ObservationModel> <
    Model<StateSpaceVariate<ParameterVariate,StateVariate,
    ObservationVariate>> {
  p:ParameterModel;
  m:InitialModel;
  f:TransitionModel;
  g:ObservationModel;

  fiber simulate(v:Variate) -> Real {
    auto xs <- v.x.walk();
    auto ys <- v.y.walk();
    
    /* parameter */
    yield sum(p.simulate(v.θ));
    
    /* initial state and initial observation */
    x:StateVariate;
    y:ObservationVariate;
    if (xs?) {
      x <- xs!;
    }
    if (ys?) {
      y <- ys!;
    }
    yield sum(m.simulate(x, v.θ)) + sum(g.simulate(y, x, v.θ));
    
    /* transition and observation */
    while (true) {
      x0:StateVariate <- x;
      if (xs?) {
        x <- xs!;
      } else {
        o:StateVariate;
        x <- o;
      }
      if (ys?) {
        y <- ys!;
      } else {
        o:ObservationVariate;
        y <- o;
      }
      yield sum(f.simulate(x, x0, v.θ)) + sum(g.simulate(y, x, v.θ));
    }
  }
}
