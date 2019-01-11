/**
 * Markov model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{0:T}) =
 *   p(\mathrm{d}\theta) p(\mathrm{d}x_0 \mid \theta) 
 *   \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1}, \theta).$$
 *
 * <center>
 * ![Graphical model depicting MarkovModel.](../figs/MarkovModel.svg)
 * </center>
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
    f:State! <- this.x.walk();
    x:State?;  // previous state
    x':State?;  // current state

    yield sum(parameter(θ));
    while (true) {
      if (f?) {  // is the next state given?
        x' <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x' <- o;
      }
      if (x?) {
        yield sum(transition(x'!, x!, θ));
      } else {
        yield sum(initial(x'!, θ));
      }
      x <- x';
    }
  }

  function start() -> Real {
    return sum(parameter(θ));
  }

  function step() -> Real {
    w:Real <- 0.0;    
    x':State;
    if (x.empty()) {
      w <- sum(initial(x', θ));
    } else {
      w <- sum(transition(x', x.back(), θ));
    }
    this.x.pushBack(x');
    return w;
  }

  function checkpoints() -> Integer? {
    /* one checkpoint for the parameters, then one for each time */
    return 1 + x.size();
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

  /**
   * Parameter proposal.
   */
  fiber parameterProposal(θ':Parameter, θ:Parameter) -> Real {
    parameter(θ');
  }
  
  /**
   * Initial proposal.
   */
  fiber initialProposal(x':State, x:State, θ:Parameter) -> Real {
    initial(x', θ);
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
