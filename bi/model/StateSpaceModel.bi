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
class StateSpaceModel<ParameterVariate,StateVariate,ObservationVariate>(
    p:@(ParameterVariate) -> Real!,
    m:@(StateVariate, ParameterVariate) -> Real!,
    f:@(StateVariate, StateVariate, ParameterVariate) -> Real!,
    g:@(ObservationVariate, StateVariate, ParameterVariate) -> Real!) <
    Model<StateSpaceVariate<ParameterVariate,StateVariate,
    ObservationVariate>> {
  /**
   * Parameter model.
   */
  p:@(ParameterVariate) -> Real! <- p;
  
  /**
   * Initial model.
   */
  m:@(StateVariate, ParameterVariate) -> Real! <- m;
  
  /**
   * Transition model.
   */
  f:@(StateVariate, StateVariate, ParameterVariate) -> Real! <- f;
  
  /**
   * Observation model.
   */
  g:@(ObservationVariate, StateVariate, ParameterVariate) -> Real! <- g;

  fiber simulate(v:Variate) -> Real {
    auto xs <- v.x.walk();
    auto ys <- v.y.walk();
    
    /* parameter */
    auto θ <- v.θ;
    yield sum(p(θ));
    
    /* initial state and initial observation */
    x:StateVariate;
    y:ObservationVariate;
    if (xs?) {
      x <- xs!;
    }
    if (ys?) {
      y <- ys!;
    }
    yield sum(m(x, θ)) + sum(g(y, x, θ));
    
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
      yield sum(f(x, x0, θ)) + sum(g(y, x, θ));
    }
  }
}
