/**
 * Linear-nonlinear state-space model. The delayed sampling feature of Birch
 * results in a Rao--Blackwellized particle filter with locally-optimal
 * proposal being applied to this model.
 *
 * Run with:
 *
 *     birch sample \
 *       --model MixedGaussianModel \
 *       --input-file input/mixed_gaussian.json \
 *       --output-file output/mixed_gaussian.json \
 *       --nparticles 256 \
 *       --ncheckpoints 50
 *
 * The model is detailed in [Lindsten and Schön (2010)](/index.md#references).
 */
class MixedGaussianModel = MarkovModel<MixedGaussianState,
    MixedGaussianParameter>;
