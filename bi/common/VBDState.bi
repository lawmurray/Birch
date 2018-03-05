/**
 * Vector-borne disease model state.
 */
class VBDState < State {
  h:SEIRState;  // humans
  m:SEIRState;  // mosquitos

  fiber simulate(θ:VBDParameter) -> Real! {
    h.simulate(θ.h);
    m.simulate(θ.m);
  }
  
  fiber simulate(x:VBDState, θ:VBDParameter) -> Real! {
    nhe:Integer;
    nme:Integer;
    
    nhe <~ Binomial(x.h.s, 1.0 - exp(-x.m.i/Real(x.h.n)));
    nme <~ Binomial(x.m.s, 1.0 - exp(-x.h.i/Real(x.h.n)));
    
    h.simulate(x.h, θ.h, nhe, x.h.e, x.h.i);
    m.simulate(x.m, θ.m, nme, x.m.e, x.m.i);
  }

  function output(writer:Writer) {
    h.output(writer.setObject("h"));
    m.output(writer.setObject("m"));
  }
}
