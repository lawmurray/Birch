/**
 * Particle Gibbs sampler.
 */
class ParticleGibbs < ConditionalParticleFilter {  
  function start() {
    if x'? {
      /* compute the conditional distribution over the start, given the
       * steps */
      auto x <- clone<ForwardModel>(archetype!);
      auto h <- x'!.getHandler();
      
      h.rewind();
      h.setDiscard(true);
      x.setHandler(h);
      x.start();
      h.setDiscard(false);
      for auto t in 1..T {
        x.step();
      }

      /* simulate the conditional distribution over the start, given the
       * steps */
      x <- clone<ForwardModel>(archetype!);
      h <- clone<EventHandler>(x'!.getHandler());
      
      h.rewind();
      h.setDelay(false);
      x.setHandler(h);
      x.start();
      h.setDelay(true);

      /* clone to all particles */
      auto replay <- x.getHandler().takeReplay();
      for auto n in 1..N {
        this.x[n] <- clone<ForwardModel>(x);
      }
      this.x[N].getHandler().setReplay(replay);
    } else {
      super.start();
    }
  }
}
