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
   * State history.
   */
  x:List<State>;
  
  /**
   * State future.
   */
  x1:List<State>;
  
  fiber simulate() -> Real {
    yield start();
    while (true) {
      yield step();
    }
  }

  function start() -> Real {
    /* parameter */
    auto θ <- θ;
    auto w <- sum(parameter(θ));

    /* initial state */
    if (!x1.empty()) {
      x':State <- x1.front();
      x1.popFront();
      w <- w + sum(initial(x', θ));
      x.pushBack(x');
    } else {
      x':State;
      w <- w + sum(initial(x', θ));
      x.pushBack(x');
    }
    
    return w;
  }

  function step() -> Real {
    auto θ <- θ;
    auto w <- 0.0;

    /* transition */
    if (!x1.empty()) {
      x':State <- x1.front();
      x1.popFront();
      w <- w + sum(transition(x', x.back(), θ));
      x.pushBack(x');
    } else {
      x':State;
      w <- w + sum(transition(x', x.back(), θ));
      x.pushBack(x');
    }

    return w;
  }

  function read(reader:Reader) {
    θ.read(reader.getObject("θ"));
    x1.read(reader.getObject("x"));
  }
  
  function write(writer:Writer) {
    θ.write(writer.setObject("θ"));
    x.write(writer.setObject("x"));
  }
  
  /**
   * Parameter model.
   */
  fiber parameter(θ':Parameter) -> Real {
    //
  }
  
  /**
   * Initial model.
   */
  fiber initial(x':State, θ:Parameter) -> Real {
    //
  }
  
  /**
   * Transition model.
   */
  fiber transition(x':State, x:State, θ:Parameter) -> Real {
    //
  }
}
