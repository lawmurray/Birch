/**
 * Vector-bourne disease model state.
 */
class VBDState {
  h:SEIRState;  // humans
  m:SEIRState;  // mosquitos

  /**
   * Initial state.
   */
  fiber run(θ:VBDParameter) -> Real! {
    h.run(θ.h);
    m.run(θ.m);
  }
  
  /**
   * Next state.
   */
  fiber run(x:VBDState, θ:VBDParameter) -> Real! {
    if (x.m.n > 0) {
      h.run(x.h, θ.h, x.h.s*x.m.i/x.m.n, x.h.e, x.h.i);
    } else {
      h.run(x.h, θ.h, 0, x.h.e, x.h.i);
    }
    if (x.h.n > 0) {
      m.run(x.m, θ.m, x.m.s*x.h.i/x.h.n, x.m.e, x.m.i);
    } else {
      m.run(x.m, θ.m, 0, x.m.e, x.m.i);
    }
  }
}
