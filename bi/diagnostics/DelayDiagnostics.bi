import basic;
import io;
import delay.Delay;

/**
 * Global diagnostics handler for delayed sampling.
 */
delayDiagnostics:DelayDiagnostics?;

/**
 * Outputs graphical representations of the delayed sampling state for
 * diagnostic purposes.
 *
 *   - N : maximum number of variates
 */
class DelayDiagnostics(N:Integer) {
  /**
   * Registered variates.
   */
  nodes:Delay?[N];
  
  /**
   * Names associated with the variates.
   */
  names:String[N];
  
  /**
   * $x$-coordinates of the variates.
   */
  xs:Integer[N];
  
  /**
   * $y$-coordinates of the variates.
   */
  ys:Integer[N];
  
  /**
   * Number of variates that have been registered.
   */
  n:Integer <- 0;
  
  /**
   * Number of events that have been triggered.
   */
  nevents:Integer <- 0;
  
  /**
   * Register a new variate. This is a callback function typically called
   * within the Delay class itself.
   *
   * *Returns* an assigned to the variate.
   */
  function register(o:Delay) -> Integer {
    assert(n < N); // otherwise no room left
    n <- n + 1;
    nodes[n] <- o;
    return n;
  }
  
  /**
   * Set the name of a node. This may be called before the node has been
   * registered; it simply associates a name with an id for whenever it is
   * used.
   *
   *   - id   : Id of the node.
   *   - name : The name.
   */
  function name(id:Integer, name:String) {
    assert n <= N;
    names[id] <- name;
  }
  
  /**
   * Set the position of a previously-registered node. This may be called
   * before the node has been registered; it simply associates a name with an
   * id for whenever it is used.
   *
   *   - id : Id of the node.
   *   - x  : $x$-coordinate.
   *   - y  : $y$-coordinate.
   *
   * A zero $x$ or $y$ coordinate suggests that an automatic layout should be
   * used for this node. This is the default anyway.
   */
  function position(id:Integer, x:Integer, y:Integer) {
    assert n <= N;
    xs[id] <- x;
    ys[id] <- y;
  }
  
  /**
   * Trigger an event.
   */
  function trigger() {
    nevents <- nevents + 1;
    dot();
  }
  
  /**
   * Output a dot graph of the current state.
   */
  function dot() {
    out:FileOutputStream("diagnostics/state" + nevents + ".dot");
    out.print("digraph {\n");
    out.print("}\n");
    out.close();
  }
}
