/**
 * Vector-borne disease model state.
 */
class VBDState < State {
  h:SEIRState;  // humans
  m:SEIRState;  // mosquitos

  fiber initial(θ:VBDParameter) -> Real {
    h.initial(θ.h);
    m.initial(θ.m);
  }
  
  fiber transition(x:VBDState, θ:VBDParameter) -> Real {
    nhe:Integer;
    nme:Integer;
    
    nhe <~ Binomial(x.h.s, 1.0 - exp(-x.m.i/Real(x.h.n)));
    nme <~ Binomial(x.m.s, 1.0 - exp(-x.h.i/Real(x.h.n)));
    
    h.transition(x.h, θ.h, nhe, x.h.e, x.h.i);
    m.transition(x.m, θ.m, nme, x.m.e, x.m.i);
  }

  function output(writer:Writer) {
    h.output(writer.setObject("h"));
    m.output(writer.setObject("m"));
  }
}
