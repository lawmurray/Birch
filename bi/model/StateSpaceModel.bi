/**
 * State-space model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{0:T}, \mathrm{d}y_{0:T}) =
 *   p(\mathrm{d}\theta) p(\mathrm{d}x_0 \mid \theta)  p(\mathrm{d}y_0
 *   \mid x_0, \theta) \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1},
 *   \theta) p(\mathrm{d}y_t \mid x_t, \theta)$$
 *
 * <center>
 * ![Graphical model depicting StateSpaceModel.](../figs/StateSpaceModel.svg)
 * </center>
 */
class StateSpaceModel<Parameter,State,Observation> <
    MarkovModel<Parameter,State> {
  /**
   * Observations.
   */
  y:List<Observation>;

  fiber simulate() -> Real {
    f:State! <- this.x.walk();
    g:Observation! <- this.y.walk();
    
    x:State?;  // previous state
    x':State?;  // current state
    y':Observation?;  // current observation

    yield start();
    while (true) {
      w:Real <- 0.0;
      if (f?) {  // is the next state given?
        x' <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x' <- o;
      }
      if (x?) {
        w <- sum(transition(x'!, x!, θ));
      } else {
        w <- sum(initial(x'!, θ));
      }
      x <- x';
      
      if (g?) {  // is the next observation given?
        y' <- g!;
      } else {
        o:Observation;
        this.y.pushBack(o);
        y' <- o;
      }
      w <- w + sum(observation(y'!, x'!, θ));
      yield w;
    }
  }

  function step() -> Real {
    assert x.size() == y.size();
    w:Real <- 0.0;
    
    x':State;
    if (x.empty()) {
      w <- sum(initial(x', θ));
    } else {
      w <- sum(transition(x', x.back(), θ));
    }
    x.pushBack(x');
    
    y':Observation;
    w <- w + sum(observation(y', x.back(), θ));
    y.pushBack(y');
    
    return w;
  }

  function checkpoints() -> Integer? {
    /* one checkpoint for the parameters, then one for each time */
    return 1 + max(x.size(), y.size());
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("y", y);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("y", y);
  }
  
  /**
   * Observation model.
   */
  fiber observation(y:Observation, x:State, θ:Parameter) -> Real {
    //
  }
}
