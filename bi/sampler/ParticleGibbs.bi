/**
 * Particle Gibbs sampler.
 */
class ParticleGibbs < ConditionalParticleFilter {  
  function start() {
    /* Step 1: Gibbs update of the start (= parameters) of the reference
     * path */
    
    /* advance through the start (= parameters) with delayed sampling, to
     * compute the prior distribution */
    x'.setSkipHandler();
    x'.start();
    
    /* replay the steps (= state) with delayed sampling, so as to compute the
     * posterior distribution over the start (= parameters) given the steps
     * (= state) and observations */
    x.setHandler(clone<Handler>(x'.getHandler()));
    for auto t in 1..T {
      x.step();
    }
    x' <- x;
    
    /* Step 2 */
    x <- clone<ForwardModel>(archetype);
    x.setHandler(x'.getHandler());
    x.start();
  }
}
