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
 *   - N : Maximum number of nodes
 *
 * To use, before running any code that uses delayed sampling, construct a
 * `DelayDiagnostic` object with sufficient capacity to hold all nodes.
 * Then, use `name()` to give a name to each node that will be of interest
 * to output, and optionally `position()` to give an explicit position.
 * Variates that are not named are not output.
 *
 * See the Birch example programs, e.g. `delay_triplet` and `delay_kalman`
 * for an example of how this is done.
 *
 * On each event, will output a `*.dot` file into `diagnostics/stateN.dot`,
 * where `N` is the event number. If positions have been explicitly given
 * using `position()`, it is recommended that these are compiled with
 * `neato`, otherwise, if positions have not been explicitly given so that
 * automatic layout is desired, use `dot`, e.g.
 *
 *     dot -Tpdf diagnostics/state1.dot > diagnostics/state1.pdf
 */
class DelayDiagnostics(N:Integer) {
  /**
   * Registered nodes.
   */
  nodes:Delay?[N];
  
  /**
   * Names of the nodes.
   */
  names:String?[N];
  
  /**
   * $x$-coordinates of the nodes.
   */
  xs:Integer?[N];
  
  /**
   * $y$-coordinates of the nodes.
   */
  ys:Integer?[N];
  
  /**
   * Number of nodes that have been registered.
   */
  n:Integer <- 0;
  
  /**
   * Number of events that have been triggered.
   */
  nevents:Integer <- 0;
  
  /**
   * Register a new node. This is a callback function typically called
   * within the Delay class itself.
   *
   * Returns an id assigned to the node.
   */
  function register(o:Delay) -> Integer {
    assert(n < N); // otherwise no room left
    n <- n + 1;
    nodes[n] <- o;
    return n;
  }
  
  /**
   * Set the name of a node.
   *
   *   - id   : Id of the node.
   *   - name : The name.
   */
  function name(id:Integer, name:String) {
    assert n <= N;
    names[id] <- name;
  }
  
  /**
   * Set the position of a node.
   *
   *   - id : Id of the node.
   *   - x  : $x$-coordinate.
   *   - y  : $y$-coordinate.
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
    out.print("  node [shape=circle]\n");
    out.print("\n");
    i:Integer;
    
    for (i in 1..n) {
      assert nodes[i]?;
      node:Delay <- nodes[i]!;
      
      if (names[i]?) {
        /* node */
        out.print("  X" + node.id + " [");
        if (node.isInitialized()) {
          out.print("style=solid penwidth=0.5 fillcolor=white fontcolor=gray color=gray fontname=\"times\"");
        } else if (node.isMarginalized()) {
          out.print("style=solid penwidth=2 fillcolor=white fontname=\"times bold\"");
        } else if (node.isRealized()) {
          out.print("style=filled penwidth=1 fillcolor=black fontcolor=white fontname=\"times\"");
        }
        if (names[i]?) {
          out.print(" label=\"" + names[i]! + "\"");
        }
        if (xs[i]? && ys[i]?) {
          out.print(" pos=\"" + xs[i]! + "," + ys[i]! + "!\"");
        }
        out.print("]\n");
      
        /* edge */
        if (node.parent?) {
          parent:Delay <- node.parent!;
          out.print("  X" + parent.id + " -> X" + node.id + " ["); 
          if (node.isInitialized()) {
            out.print("style=solid arrowhead=empty penwidth=0.5 color=gray");
          } else if (node.isMarginalized() && parent.isMarginalized()) {
            out.print("style=solid penwidth=2");
          } else {
            out.print("style=solid penwidth=1");
          }
          out.print("]\n");
        }
      }
    }
    
    out.print("}\n");
    out.close();
  }
}
