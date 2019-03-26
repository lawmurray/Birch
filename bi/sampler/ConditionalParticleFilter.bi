/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter < ParticleFilter {   
  function resample() {
    if fN.empty() {
      a <- multinomial_ancestors(w);
    } else {
      a <- multinomial_conditional_ancestors(w);
    }
    w <- vector(0.0, nparticles);
  }
}
