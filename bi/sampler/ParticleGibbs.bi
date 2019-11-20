/**
 * Particle Gibbs sampler. This behaves as per ParticleFilter for the
 * first sample. For subsequent samples it conditions on a particle drawn
 * from the previous iteration, while additionally performing a Gibbs update
 * of parameters conditioned on the same particle.
 */
class ParticleGibbs < ConditionalParticleFilter {
  function start() {
    if h'? {
      /* reference particle, so on a subsequent iteration; for the reference
       * trajectory, first replay it with the start (parameters) marginalized
       * out, so as to compute the distribution over them conditioned on the
       * steps (states) */
      x[b].h.setMode(SKIP_DELAY);
      x[b].start();
      x[b].h.setMode(REPLAY_DELAY);
      for t in 1..T {
        x[b].step();
      }
      h' <- x[b].h;
      h'!.rewind();

      /* now replay the start (parameters) of the reference trajectory, in
       * immediate sampling mode, so sample new parameters from that
       * conditional distribution */
      x[b] <- clone<ForwardModel>(archetype!);
      x[b].h.setRecord(true);
      x[b].h.trace.forwardCount <- h'!.trace.forwardCount;
      x[b].h.trace.forward <- h'!.trace.takeForward();
      x[b].h.setMode(REPLAY_IMMEDIATE);
      x[b].start();

      /* clone to all particles */
      auto forwardCount <- x[b].h.trace.forwardCount;
      auto forward <- x[b].h.trace.takeForward();
      x[b].h.setMode(PLAY_DELAY);
      for n in 1..N {
        this.x[n] <- clone<ForwardModel>(x[b]);
      }
      x[b].h.setMode(REPLAY_DELAY);
      x[b].h.trace.putForward(forward, forwardCount);
    } else {
      /* no reference particle, so on the first iteration, do as normal */
      super.start();
    }
  }
}
