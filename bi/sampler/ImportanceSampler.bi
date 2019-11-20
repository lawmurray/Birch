/**
 * Importance sampler.
 */
class ImportanceSampler < Sampler {
  fiber sample(model:Model) -> (Model, Real) {
    while true {
      auto x <- clone<Model>(model);
      auto w <- delay.handle(x.simulate());
      yield (x, w);
    }
  }
}
