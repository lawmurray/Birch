/**
 * Particle Gibbs sampler. This behaves as per ParticleFilter for the
 * first sample. For subsequent samples it conditions on a particle drawn
 * from the previous iteration, while additionally performing a Gibbs update
 * of parameters conditioned on the same particle.
 */
class ParticleGibbs < ConditionalParticleFilter {
  function start() {
    if !x'? {
      /* no reference particle, so on the first iteration, do as normal */
      super.start();
    } else {
      /* reference particle, so on a subsequent iteration; for the reference
       * trajectory, first replay it with the start (parameters) marginalized
       * out, so as to compute the distribution over them conditioned on the
       * steps (states) */
      auto h <- clone<EventHandler>(x'!.getHandler());
      auto x <- clone<ForwardModel>(archetype!);
      x.setHandler(h);
      h.rewind();
      h.setMode(SKIP_DELAY);
      x.start();
      h.setMode(REPLAY_DELAY);
      for auto t in 1..T {
        x.step();
      }

      /* now replay the start (parameters) of the reference trajectory, in
       * immediate sampling mode, so sample new parameters from that
       * conditional distribution */
      x <- clone<ForwardModel>(archetype!);
      x.setHandler(h);
      h.rewind();
      h.setMode(REPLAY_IMMEDIATE);
      x.start();

      /* clone to all particles */
      auto forward <- h.trace.forward;
      h.trace.forward <- nil;
      h.setMode(PLAY_DELAY);
      
      for auto n in 1..N {
        this.x[n] <- clone<ForwardModel>(x);
      }
      this.x[b].getHandler().setMode(REPLAY_DELAY);
      this.x[b].getHandler().trace.forward <- forward;
    }
  }
}
