/**
 * Particle marginal Metropolis--Hastings sampler.
 */
class ParticleMarginalMetropolisHastings < Sampler {
  /**
   * Inner sampler.
   */
  sampler:Sampler?;
  
  fiber sample(m:Model) -> (Model, Real) {
    sampler!.sample(m);
  }

  function read(buffer:Buffer) {
    super.read(buffer);

    /* create the inner sampler */
    className:String?;
    samplerBuffer:Buffer? <- buffer.getObject("sampler");
    if (samplerBuffer?) {
      className <- samplerBuffer!.getString("class");
    }
    if (!className?) {
      className <- "ParticleFilter";
    }
    sampler <- Sampler?(make(className!));
    if (!sampler?) {
      error(className! + " is not a subtype of Sampler.");
    }
    buffer.get("sampler", sampler!);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("sampler", sampler!);
  }
}
