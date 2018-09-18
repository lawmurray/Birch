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
class StateSpaceModel<Parameter,State,Observation>(
    p:@(Parameter) -> Real!,
    m:@(State, Parameter) -> Real!,
    f:@(State, State, Parameter) -> Real!,
    g:@(Observation, State, Parameter) -> Real!) < Model {
  /**
   * Variate.
   */
  v:StateSpaceVariate<Parameter,State,Observation>;
    
  /**
   * Parameter model.
   */
  p:@(Parameter) -> Real! <- p;
  
  /**
   * Initial model.
   */
  m:@(State, Parameter) -> Real! <- m;
  
  /**
   * Transition model.
   */
  f:@(State, State, Parameter) -> Real! <- f;
  
  /**
   * Observation model.
   */
  g:@(Observation, State, Parameter) -> Real! <- g;

  fiber simulate() -> Real {
    /* parameter */
    auto θ <- v.θ;
    yield sum(p(θ));

    auto xs <- v.x.walk();
    auto ys <- v.y.walk();    
    
    /* initial state and initial observation */
    x:State;
    y:Observation;
    
    if (xs?) {
      x <- xs!;
    }
    if (ys?) {
      y <- ys!;
    }
    yield sum(m(x, θ)) + sum(g(y, x, θ));
    
    /* transition and observation */
    while (true) {
      x0:State <- x;
      if (xs?) {
        x <- xs!;
      } else {
        x':State;
        x <- x';
        v.x.pushBack(x);
      }
      if (ys?) {
        y <- ys!;
      } else {
        y':Observation;
        y <- y';
        v.y.pushBack(y);
      }
      yield sum(f(x, x0, θ)) + sum(g(y, x, θ));
    }
  }

  function read(reader:Reader) {
    v.read(reader);
  }
  
  function write(writer:Writer) {
    v.write(writer);
  }
}
