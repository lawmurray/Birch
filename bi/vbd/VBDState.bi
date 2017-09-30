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
    /* The total number of blood meals is Poisson distributed with rate given
     * by the total population of mosquitos. We then solve the occupancy
     * problems: how many susceptible humans are bitten by at least one
     * infected mosquito, and how many susceptible mosquitoes bite at least
     * one infected person. The Poisson distribution makes this independent
     * for each individual; consider how one could do an independent Poisson
     * draw for each mosquito then categorical draws for the humans that it
     * bites, but how this computation can be simplified */

    rate:Real <- x.m.n; // rate for Poisson number of bites
     
    nhe:Integer <- simulate_binomial(x.h.s, 1.0 - exp(-rate*x.m.i/(x.m.n*x.h.n)));
    h.run(x.h, θ.h, nhe, x.h.e, x.h.i);
    
    nme:Integer <- simulate_binomial(x.m.s, 1.0 - exp(-rate*x.h.i/(x.h.n*x.m.n)));
    m.run(x.m, θ.m, nme, x.m.e, x.m.i);
  }

  function output(prefix:String) {
    h.output(prefix + "h.");
    m.output(prefix + "m.");
  }
}
