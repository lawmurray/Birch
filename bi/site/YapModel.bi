/**
 * Model for Yap case study.
 *
 *   - T: number of time steps.
 */
class YapModel(T:Integer) {
  θ:YapParameter;
  x:YapState[T];
  
  /**
   * Run the model.
   */
  fiber run() -> Real! {
    /* read observations */
    input();
    
    /* simulate */
    θ.run();
    x[1].run(θ);
    for (t:Integer in 2..T) {
      x[t].run(x[t-1], θ);
    }
    
    /* output */
    output();
  }
  
  function input() {
    input:FileInputStream("data/yap_dengue.csv");
    
    t:Integer <- input.readInteger();
    y:Integer <- input.readInteger();
    while (t <= T && !input.eof()) {
      x[t].y <- y;
      t <- input.readInteger();
      y <- input.readInteger();
    }
  }
  
  function output() {
    for (t:Integer in 1..T) {
      stdout.print(x[t].h.s + ",");
      stdout.print(x[t].h.e + ",");
      stdout.print(x[t].h.i + ",");
      stdout.print(x[t].h.r + "\n");
    }
  }
}
