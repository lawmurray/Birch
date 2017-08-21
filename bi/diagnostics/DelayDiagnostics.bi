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
   * Number of graphs that have been output.
   */
  noutputs:Integer <- 0;
  
  /**
   * Register a new node. This is a callback function typically called
   * within the Delay class itself.
   *
   * Returns an id assigned to the node.
   */
  function register(o:Delay) -> Integer {
    assert n < N; // otherwise no room left
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
    /* avoid outputting empty graphs, so check first that at least one
     * registered node has been named */
    i:Integer <- 1;
    isEmpty:Boolean <- true;
    while (i <= n && isEmpty) {
      isEmpty <- !names[i]?;
      i <- i + 1;
    }
    if (!isEmpty) {
      noutputs <- noutputs + 1;
      dot();
    }
  }
  
  /**
   * Output a dot graph of the current state.
   */
  function dot() {
    /* pad file name with zeros */
    Z:Integer <- 8 - Integer(ceil(log10(Real(noutputs + 1))));
    z:Integer;
    filename:String <- "diagnostics/state";
    for (z in 1..Z) {
      filename <- filename + "0";
    }
    filename <- filename + noutputs + ".dot";
    
    /* open file */
    out:FileOutputStream(filename);
    
    /* output dot graph */
    out.print("digraph {\n");
    out.print("  node [shape=circle]\n");
    out.print("\n");
    i:Integer;
    
    for (i in 1..N) {
      if (nodes[i]? && names[i]?) {
        node:Delay <- nodes[i]!;
      
        /* output node */
        out.print("  X" + node.id + " [");
        if (node.isInitialized()) {
          out.print("style=solid fillcolor=white fontcolor=gray color=gray margin=\"0.04,0.02\"");
        } else if (node.isMarginalized()) {
          out.print("style=solid fillcolor=white margin=\"0.04,0.02\"");
        } else if (node.isRealized()) {
          out.print("style=filled fillcolor=black fontcolor=white margin=\"0.04,0.02\"");
        } else {
          assert false;
        }
        out.print(" label=\"" + names[i]! + "\"");
        if (xs[i]? && ys[i]?) {
          out.print(" pos=\"" + xs[i]! + "," + ys[i]! + "!\"");
        }
        out.print("]\n");
      
        /* output edge */
        if (node.parent?) {
          parent:Delay <- node.parent!;
          out.print("  X" + parent.id + " -> X" + node.id + " ["); 
          if (node.isInitialized()) {
            out.print("style=solid penwidth=1 color=gray");
          } else if (node.isMarginalized() && parent.isMarginalized()) {
            out.print("style=solid penwidth=1");
          } else {
            out.print("style=solid penwidth=1");
          }
          out.print("]\n");
        }
      } else if (xs[i]? && ys[i]?) {
        /* output an invisible node to preserve this position and the overall
         * size of the graph between outputs */
        out.print("  Z" + i + " [style=invis pos=\"" + xs[i]! + "," + ys[i]! + "!\"]\n");
      }
    }
    
    out.print("}\n");
    out.close();
  }
}
