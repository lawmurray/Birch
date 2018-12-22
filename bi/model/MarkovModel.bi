/**
 * Markov model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{0:T}) =
 *   p(\mathrm{d}\theta) p(\mathrm{d}x_0 \mid \theta) 
 *   \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1}, \theta).$$
 */
class MarkovModel<Parameter,State> < Model {
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * States.
   */
  x:List<State>;

  fiber simulate() -> Real {
    /* iterate through given times (those with clamped values)  */
    x':State! <- this.x.walk();
    x:State?;
    while (x'?) {
      if (x?) {
        yield sum(transition(x'!, x!, θ));
      } else {
        yield sum(parameter(θ)) + sum(initial(x'!, θ));
      }
      x <- x'!;
    }
    
    /* indefinitely generate future states */
    while (true) {
      x':State;
      this.x.pushBack(x');
      
      if (x?) {
        yield sum(transition(x', x!, θ));
      } else {
        yield sum(parameter(θ)) + sum(initial(x', θ));
      }
      x <- x';
    }
  }

  function checkpoints() -> Integer? {
    if (x.empty()) {
      return nil;
    } else {
      return x.size();
    }
  }
     
  /**
   * Parameter model.
   */
  fiber parameter(θ:Parameter) -> Real {
    //
  }
  
  /**
   * Initial model.
   */
  fiber initial(x:State, θ:Parameter) -> Real {
    //
  }
  
  /**
   * Transition model.
   */
  fiber transition(x':State, x:State, θ:Parameter) -> Real {
    //
  }

  function read(buffer:Buffer) {
    buffer.get("θ", θ);
    buffer.get("x", x);
  }
  
  function write(buffer:Buffer) {
    buffer.set("θ", θ);
    buffer.set("x", x);
  }
}
