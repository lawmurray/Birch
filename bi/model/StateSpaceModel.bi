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
 *
 * A model inheriting from `StateSpaceModel` overrides the `parameter`,
 * `initial`, `transition` and `observation` member fibers to specify the
 * individual components of the joint distribution, rather than the
 * `simulate` member fiber; likewise for the `propose` analogs.
 *
 * As `MarkovModel`, from which it inherits, `StateSpaceModel` provides an
 * alternative function-based interface based on the `start` and `stop`
 * member functions.
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
    
    u:State?;        // previous state
    x:State?;        // current state
    y:Observation?;  // current observation

    yield sum(parameter(θ));
    while true {
      w:Real <- 0.0;
      if f? {  // is the next state given?
        x <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x <- o;
      }
      if x? {
        w <- sum(transition(x!, u!, θ));
      } else {
        w <- sum(initial(x!, θ));
      }
      u <- x;
      
      if g? {  // is the next observation given?
        y <- g!;
      } else {
        o:Observation;
        this.y.pushBack(o);
        y <- o;
      }
      w <- w + sum(observation(y!, x!, θ));
      yield w;
    }
  }

  fiber propose() -> Real {
    f:State! <- this.x.walk();
    g:Observation! <- this.y.walk();
    
    u:State?;        // previous state
    x:State?;        // current state
    y:Observation?;  // current observation

    yield sum(proposeParameter(θ));
    while true {
      w:Real <- 0.0;
      if f? {  // is the next state given?
        x <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x <- o;
      }
      if x? {
        w <- sum(proposeTransition(x!, u!, θ));
      } else {
        w <- sum(proposeInitial(x!, θ));
      }
      u <- x;
      
      if g? {  // is the next observation given?
        y <- g!;
      } else {
        o:Observation;
        this.y.pushBack(o);
        y <- o;
      }
      w <- w + sum(proposeObservation(y!, x!, θ));
      yield w;
    }
  }

  fiber propose(m:Model) -> Real {
    auto n <- StateSpaceModel<Parameter,State,Observation>?(m);
    if n? {
      propose(n!);
    } else {
      error("previous state has incorrect type");
    }
  }
  
  fiber propose(m:StateSpaceModel<Parameter,State,Observation>) -> Real {  
    auto θ <- m.θ;
    auto θ' <- this.θ;
    auto f <- m.x.walk(); 
    auto f' <- this.x.walk();
    auto g <- m.y.walk(); 
    auto g' <- this.y.walk();
    
    u:State?;         // previous state of m
    x:State?;         // current state of m
    y:Observation?;   // current observation of m
    u':State?;        // previous state of this
    x':State?;        // current state of this
    y':Observation?;  // current observation of this

    yield sum(proposeParameter(θ', θ));
    while true {
      w:Real <- 0.0;
      if f? {
        x <- f!;
      } else {
        error("previous state has incorrect number of checkpoints");
      }
      if f'? {  // is the next state given?
        x' <- f'!;
      } else {
        o:State;
        this.x.pushBack(o);
        x' <- o;
      }
      if u'? {
        w <- w + sum(proposeTransition(x'!, u'!, θ', x!, u!, θ));
      } else {
        w <- w + sum(proposeInitial(x'!, θ', x!, θ));
      }
      u <- x;
      u' <- x';

      if g? {
        y <- g!;
      } else {
        error("previous state has incorrect number of checkpoints");
      }
      if g'? {  // is the next observation given?
        y' <- g'!;
      } else {
        o:Observation;
        this.y.pushBack(o);
        y' <- o;
      }
      w <- w + sum(proposeObservation(y'!, x'!, θ', y!, x!, θ));
      yield w;
    }
  }

  function step() -> Real {
    assert x.size() == y.size();
    w:Real <- 0.0;
    
    x:State;
    if this.x.empty() {
      w <- sum(initial(x, θ));
    } else {
      w <- sum(transition(x, this.x.back(), θ));
    }
    this.x.pushBack(x);
    
    y:Observation;
    w <- w + sum(observation(y, this.x.back(), θ));
    this.y.pushBack(y);
    
    return w;
  }

  function checkpoints() -> Integer? {
    /* one checkpoint for the parameters, then one for each time */
    return 1 + max(x.size(), y.size());
  }

  /**
   * Observation model.
   *
   * - y: The observations, to be set.
   * - x: The current state.
   * - θ: The parameters.
   */
  fiber observation(y:Observation, x:State, θ:Parameter) -> Real {
    //
  }

  /**
   * Observation proposal.
   *
   * - y: The observations, to be set.
   * - x: The current state.
   * - θ: The parameters.
   *
   * By default calls `observation(y, x, θ)`.
   */
  fiber proposeObservation(y:Observation, x:State, θ:Parameter) -> Real {
    observation(y, x, θ);
  }

  /**
   * Observation proposal.
   *
   * - y': The observations, to be set.
   * - x': The current state.
   * - θ': The parameters.
   * - y: The last observations.
   * - x: The last current state.
   * - θ: The last parameters.
   *
   * By default calls `proposeObservation(y', x', θ')`.
   */
  fiber proposeObservation(y':Observation, x':State, θ':Parameter,
      y:Observation, x:State, θ:Parameter) -> Real {
    proposeObservation(y', x', θ');
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("y", y);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("y", y);
  }
}
