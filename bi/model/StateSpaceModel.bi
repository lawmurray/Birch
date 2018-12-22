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
class StateSpaceModel<Parameter,State,Observation> <
    MarkovModel<Parameter,State> {
  /**
   * Observations.
   */
  y:List<Observation>;

  fiber simulate() -> Real {
    /* iterate through given times (those with clamped values)  */
    x':State! <- this.x.walk();
    y:Observation! <- this.y.walk();
    x:State?; // previous state
    
    while (x'? && y?) {
      if (x?) {
        yield sum(transition(x'!, x!, θ)) + sum(observation(y!, x'!, θ));
      } else {
        yield sum(parameter(θ)) + sum(initial(x'!, θ)) + sum(observation(y!, x'!, θ));
      }
      x <- x'!;
    }
    
    /* remaining times with state given but no observation given */
    while (x'?) {
      y:Observation;
      this.y.pushBack(y);
      
      if (x?) {
        yield sum(transition(x'!, x!, θ)) + sum(observation(y, x'!, θ));
      } else {
        yield sum(parameter(θ)) + sum(initial(x'!, θ)) + sum(observation(y, x'!, θ));
      }
      x <- x'!;
    }
    
    /* remaining times with observation given but no state given */
    while (y?) {
      x':State;
      this.x.pushBack(x');
      
      if (x?) {
        yield sum(transition(x', x!, θ)) + sum(observation(y!, x', θ));
      } else {
        yield sum(parameter(θ)) + sum(initial(x', θ)) + sum(observation(y!, x', θ));
      }
      x <- x';
    }
    
    /* indefinitely generate future states */
    while (true) {
      x':State;
      y:Observation;
      this.x.pushBack(x');
      this.y.pushBack(y);
      
      if (x?) {
        yield sum(transition(x', x!, θ)) + sum(observation(y, x', θ));
      } else {
        yield sum(parameter(θ)) + sum(initial(x', θ)) + sum(observation(y, x', θ));
      }
      x <- x';
    }
  }

  function checkpoints() -> Integer? {
    if (x.empty() && y.empty()) {
      return nil;
    } else {
      return max(x.size(), y.size());
    }
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
