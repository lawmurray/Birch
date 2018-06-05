/**
 * Linear-Gaussian state-space model. The delayed sampling feature of Birch
 * results in a Kalman filter being applied to this model.
 *
 * Run with:
 *
 *     birch sample \
 *       --model LinearGaussianModel \
 *       --input-file input/linear_gaussian.json \
 *       --output-file output/linear_gaussian.json \
 *       --ncheckpoints 10
 */
class LinearGaussianModel = MarkovModel<LinearGaussianState,
    LinearGaussianParameter>;
