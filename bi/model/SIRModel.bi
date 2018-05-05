/**
 * SIR (susceptible-infectious-recovered) model.
 *
 * Run the example using:
 *
 *     birch sample \
 *       --model SIRModel \
 *       --input-file input/russian_influenza.json \
 *       --output-file output/russian_influenza.json \
 *       --ncheckpoints 14 \
 *       --nparticles 100 \
 *       --nsamples 10
 *
 * The data set is of an outbreak of Russian influenza at a boy's boarding
 * school in northern England [(Anonymous, 1978)](/index.md#references). The
 * model on which this is based is described in
 * [Murray et. al. (2018)](/index.md#references).
 */
class SIRModel = MarkovModel<SIRState,SIRParameter>;
