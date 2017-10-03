/**
 * Vector-bourne disease model state.
 */
class VBDState {
  h:SEIRState;  // humans
  m:SEIRState;  // mosquitos

  fiber run(θ:VBDParameter) -> Real! {
    h.run(θ.h);
    m.run(θ.m);
  }
  
  fiber run(x:VBDState, θ:VBDParameter) -> Real! {
    nhe:Integer;
    nme:Integer;
    
    nhe <~ Binomial(x.h.s, 1.0 - exp(-x.m.i/x.h.n));
    nme <~ Binomial(x.m.s, 1.0 - exp(-x.h.i/x.h.n));
    
    h.run(x.h, θ.h, nhe, x.h.e, x.h.i);
    m.run(x.m, θ.m, nme, x.m.e, x.m.i);
  }

  function output(prefix:String) {
    h.output(prefix + "h.");
    m.output(prefix + "m.");
  }
}
