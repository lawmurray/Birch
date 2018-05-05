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
 * school in northern England[^1].
 *
 * [^1]: Anonymous (1978). Influenza in a boarding school. *British Medical
 * Journal*. **1**:587.
 */
class SIRModel = MarkovModel<SIRState,SIRParameter>;
