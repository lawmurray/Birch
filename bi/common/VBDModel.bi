/**
 *
 */
final class VBDModel < MarkovModel<VBDParameter,VBDState> {
  h:SEIRModel;
  m:SEIRModel;

  fiber parameter(θ:VBDParameter) -> Event {
    θ.h.ν <- 0.0;
    θ.h.μ <- 1.0;
    θ.h.λ ~ Beta(1.0, 1.0);
    θ.h.δ ~ Beta(1.0, 1.0);
    θ.h.γ ~ Beta(1.0, 1.0);
    
    θ.m.ν <- 1.0/7.0;
    θ.m.μ <- 6.0/7.0;
    θ.m.λ ~ Beta(1.0, 1.0);
    θ.m.δ ~ Beta(1.0, 1.0);
    θ.m.γ <- 0.0;
  }

  fiber initial(x:VBDState, θ:VBDParameter) -> Event {
    h.initial(x.h, θ.h)!!;
    m.initial(x.m, θ.m)!!;
  }
  
  fiber transition(x':VBDState, x:VBDState, θ:VBDParameter) -> Event {
    nhe:Integer;
    nme:Integer;
    
    nhe <~ Binomial(x.h.s, 1.0 - exp(-x.m.i/Real(x.h.n)));
    nme <~ Binomial(x.m.s, 1.0 - exp(-x.h.i/Real(x.h.n)));

    h.transition(x'.h, x.h, θ.h, nhe, x.h.e, x.h.i)!!;
    m.transition(x'.m, x.m, θ.m, nme, x.m.e, x.m.i)!!;
  }
}
