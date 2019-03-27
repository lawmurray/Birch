/**
 * Particle Gibbs sampler.
 */
class ParticleGibbs < ConditionalParticleFilter {  
  function start() {
    /* perform on Gibbs move on the start of the reference path */
    auto x <- clone<ForwardModel>(archetype!);
    auto h <- clone<Handler>(x'!.getHandler());
    x.setHandler(h);
    h.rewind();
    
    /* replay the start, but discard the trace so as to re-establish the
     * prior distribution over the start */
    h.setDiscard(true);
    x.start();
    h.setDiscard(false);
    
    /* replay the steps, so as to compute the conditional distribution over
     * the start given the steps */
    for auto t in 1..T {
      x.step();
    }
  }
}
