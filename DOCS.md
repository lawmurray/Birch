
# Summary

| Variable | Description |
| --- | --- |
| *delayDiagnostics:[DelayDiagnostics](#delaydiagnostics-367)?* | Global diagnostics handler for delayed sampling. |
| *inf:[Real64](#real64-214)* | $\infty$ |
| *stderr:[StdErrStream](#stderrstream-1076)* | Standard error. |
| *stdin:[StdInStream](#stdinstream-1079)* | Standard input. |
| *stdout:[StdOutStream](#stdoutstream-1082)* | Standard output. |
| *π:[Real64](#real64-214)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-380) | Create Bernoulli distribution. |
| [Beta](#beta-397) | Create Beta distribution. |
| [Binomial](#binomial-414) | Create binomial distribution. |
| [Binomial](#binomial-571) | Create beta-binomial distribution. |
| [Boolean](#boolean-3) | Convert other basic types to Boolean. |
| [Gamma](#gamma-431) | Create Gamma distribution. |
| [Gaussian](#gaussian-453) | Create Gaussian distribution. |
| [Gaussian](#gaussian-505) | Create multivariate Gaussian distribution. |
| [I](#i-1200) | Create identity matrix. |
| [Integer](#integer-24) | Convert other basic types to Integer. |
| [Integer32](#integer32-35) | Convert other basic types to Integer32. |
| [Integer64](#integer64-91) | Convert other basic types to Integer64. |
| [LogGaussian](#loggaussian-478) | Create log-Gaussian distribution. |
| [LogNormal](#lognormal-481) | Create log-Gaussian distribution. |
| [NegativeBinomial](#negativebinomial-525) | Create. |
| [Normal](#normal-456) | Create Gaussian distribution. |
| [Normal](#normal-508) | Create multivariate Gaussian distribution. |
| [Poisson](#poisson-538) | Create Poisson distribution. |
| [Poisson](#poisson-586) | Create gamma-Poisson distribution. |
| [Real](#real-147) | Convert other basic types to Real. |
| [Real32](#real32-158) | Convert other basic types to Real32. |
| [Real64](#real64-216) | Convert other basic types to Real64. |
| [String](#string-274) | Convert other basic types to String. |
| [Uniform](#uniform-555) | Create a Uniform distribution. |
| [abs](#abs-79) | Absolute value. |
| [abs](#abs-135) | Absolute value. |
| [abs](#abs-202) | Absolute value. |
| [abs](#abs-260) | Absolute value. |
| [adjacent_difference](#adjacent-difference-1468) | Inclusive prefix sum. |
| [ancestor](#ancestor-1479) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-1473) | Sample an ancestry vector for a vector of log-weights. |
| [beta](#beta-1283) | The beta function. |
| [beta](#beta-1286) | The beta function. |
| [choose](#choose-1295) | The binomial coefficient. |
| [choose](#choose-1298) | The binomial coefficient. |
| [columns](#columns-1163) | Number of columns of a matrix. |
| [columns](#columns-1165) | Number of columns of a matrix. |
| [columns](#columns-1167) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-1494) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-1506) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-1140) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-1464) | Inclusive prefix sum. |
| [fclose](#fclose-21) | Close a file. |
| [fopen](#fopen-16) | Open a file for reading. |
| [fopen](#fopen-19) | Open a file. |
| [gamma](#gamma-1274) | The gamma function. |
| [gamma](#gamma-1276) | The gamma function. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-1460) | Inclusive prefix sum. |
| [inverse](#inverse-1144) | Inverse of a matrix. |
| [isnan](#isnan-213) | Does this have the value NaN? |
| [isnan](#isnan-271) | Does this have the value NaN? |
| [lbeta](#lbeta-1289) | Logarithm of the beta function. |
| [lbeta](#lbeta-1292) | Logarithm of the beta function. |
| [lchoose](#lchoose-1301) | Logarithm of the binomial coefficient. |
| [lchoose](#lchoose-1304) | Logarithm of the binomial coefficient. |
| [length](#length-337) | Length of a string. |
| [length](#length-1414) | Length of a vector. |
| [length](#length-1416) | Length of a vector. |
| [length](#length-1418) | Length of a vector. |
| [lgamma](#lgamma-1278) | Logarithm of the gamma function. |
| [lgamma](#lgamma-1280) | Logarithm of the gamma function. |
| [llt](#llt-1147) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-1456) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-1174) | Create matrix filled with a given scalar value. |
| [matrix](#matrix-1181) | Create matrix filled with a given scalar value. |
| [matrix](#matrix-1188) | Create matrix filled with a given scalar value. |
| [max](#max-85) | Maximum of two values. |
| [max](#max-141) | Maximum of two values. |
| [max](#max-208) | Maximum of two values. |
| [max](#max-266) | Maximum of two values. |
| [max](#max-1447) | Maximum of a vector. |
| [min](#min-88) | Minimum of two values. |
| [min](#min-144) | Minimum of two values. |
| [min](#min-211) | Minimum of two values. |
| [min](#min-269) | Minimum of two values. |
| [min](#min-1451) | Minimum of a vector. |
| [mod](#mod-82) | Modulus. |
| [mod](#mod-138) | Modulus. |
| [mod](#mod-205) | Modulus. |
| [mod](#mod-263) | Modulus. |
| [norm](#norm-1136) | Norm of a vector. |
| [observe_bernoulli](#observe-bernoulli-1203) | Observe a Bernoulli variate. |
| [observe_beta](#observe-beta-1239) | Observe a beta variate. |
| [observe_beta_binomial](#observe-beta-binomial-1216) | Observe a beta-binomial variate. |
| [observe_binomial](#observe-binomial-1207) | Observe a binomial variate. |
| [observe_gamma](#observe-gamma-1235) | Observe a gamma variate. |
| [observe_gaussian](#observe-gaussian-1227) | Observe a Gaussian variate. |
| [observe_log_gaussian](#observe-log-gaussian-1231) | Observe a log-Gaussian variate. |
| [observe_negative_binomial](#observe-negative-binomial-1211) | Observe a negative binomial variate. |
| [observe_poisson](#observe-poisson-1219) | Observe a Poisson variate. |
| [observe_uniform](#observe-uniform-1223) | Observe a uniform variate. |
| [permute_ancestors](#permute-ancestors-1500) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [rows](#rows-1157) | Number of rows of a matrix. |
| [rows](#rows-1159) | Number of rows of a matrix. |
| [rows](#rows-1161) | Number of rows of a matrix. |
| [scalar](#scalar-1190) | Convert single-element matrix to scalar value. |
| [scalar](#scalar-1192) | Convert single-element matrix to scalar value. |
| [scalar](#scalar-1194) | Convert single-element matrix to scalar value. |
| [scalar](#scalar-1435) | Convert single-element vector to scalar value. |
| [scalar](#scalar-1437) | Convert single-element vector to scalar value. |
| [scalar](#scalar-1439) | Convert single-element vector to scalar value. |
| [seed](#seed-1241) | Seed the pseudorandom number generator. |
| [simulate_bernoulli](#simulate-bernoulli-1243) | Simulate a Bernoulli variate. |
| [simulate_beta](#simulate-beta-1272) | Simulate a beta variate. |
| [simulate_beta_binomial](#simulate-beta-binomial-1253) | Simulate a beta-binomial variate. |
| [simulate_binomial](#simulate-binomial-1246) | Simulate a binomial variate. |
| [simulate_gamma](#simulate-gamma-1267) | Simulate a gamma variate. |
| [simulate_gaussian](#simulate-gaussian-1261) | Simulate a Gaussian variate. |
| [simulate_log_gaussian](#simulate-log-gaussian-1264) | Simulate a log-Gaussian variate. |
| [simulate_negative_binomial](#simulate-negative-binomial-1249) | Simulate a negative binomial variate. |
| [simulate_poisson](#simulate-poisson-1255) | Simulate a Poisson variate. |
| [simulate_uniform](#simulate-uniform-1258) | Simulate a uniform variate. |
| [solve](#solve-1151) | Solve a system of equations. |
| [solve](#solve-1155) | Solve a system of equations. |
| [squaredNorm](#squarednorm-1138) | Squared norm of a vector. |
| [sum](#sum-1443) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-1486) | Systematic resampling. |
| [transpose](#transpose-1142) | Transpose of a matrix. |
| [vector](#vector-1423) | Create vector filled with a given scalar. |
| [vector](#vector-1428) | Create vector filled with a given value. |
| [vector](#vector-1433) | Create vector filled with a given value. |

| Program | Brief description |
| --- | --- |
| [build](#build-1026) | Build the project. |
| [check](#check-1027) | Check the file structure of the project for possible issues. |
| [clean](#clean-1028) | Clean the project directory of all build files. |
| [dist](#dist-1029) | Build a distributable archive for the project. |
| [docs](#docs-1030) | Build the reference documentation for the project. |
| [init](#init-1032) | Initialise the working directory for a new project. |
| [install](#install-1033) | Install the project. |
| [uninstall](#uninstall-1034) | Uninstall the project. |

| Basic Type | Brief description |
| --- | --- |
| [Boolean](#boolean-1) | A Boolean value. |
| [File](#file-14) | A file handle. |
| [Integer32](#integer32-33) | A 32-bit integer. |
| [Integer64](#integer64-89) | A 64-bit integer. |
| [Real32](#real32-156) | A 32-bit (single precision) floating point value. |
| [Real64](#real64-214) | A 64-bit (double precision) floating point value. |
| [String](#string-272) | A string value. |

| Alias Type | Brief description |
| --- | --- |
| [Integer](#integer-22) | An integer value of default type. |
| [LogNormal](#lognormal-474) | Synonym for LogGaussian. |
| [MultivariateNormal](#multivariatenormal-500) | Synonym for MultivariateGaussian. |
| [Normal](#normal-449) | Synonym for Gaussian. |
| [Real](#real-145) | A floating point value of default type. |

| Class Type | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-377) | Bernoulli distribution. |
| [Beta](#beta-393) | Beta distribution. |
| [BetaBinomial](#betabinomial-567) | Binomial with conjugate prior on success probability. |
| [Binomial](#binomial-410) | Binomial distribution. |
| [Delay](#delay-788) | Node interface for delayed sampling. |
| [DelayBoolean](#delayboolean-800) | Abstract delay variate with Boolean value. |
| [DelayDiagnostics](#delaydiagnostics-367) | Outputs graphical representations of the delayed sampling state for diagnostic purposes. |
| [DelayInteger](#delayinteger-812) | Abstract delay variate with Integer value. |
| [DelayReal](#delayreal-824) | Abstract delay variate with Real value. |
| [DelayRealVector](#delayrealvector-835) | Abstract delay variate with real vector value. |
| [FileInputStream](#fileinputstream-1037) | File input stream. |
| [FileOutputStream](#fileoutputstream-1040) | File output stream. |
| [Gamma](#gamma-427) | Gamma distribution. |
| [GammaPoisson](#gammapoisson-583) | Poisson with conjugate prior on rate. |
| [Gaussian](#gaussian-448) | Gaussian distribution. |
| [GaussianExpression](#gaussianexpression-843) | Expression that is an affine transformation of a Gaussian variable. |
| [GaussianLogExpression](#gaussianlogexpression-911) | Expression that is an affine transformation of the logarithm of a log-Gaussian variable. |
| [GaussianWithGaussianExpressionMean](#gaussianwithgaussianexpressionmean-598) | Gaussian with conjugate prior on mean. |
| [GaussianWithGaussianLogExpressionMean](#gaussianwithgaussianlogexpressionmean-617) | Gaussian with conjugate prior on mean. |
| [GaussianWithGaussianMean](#gaussianwithgaussianmean-636) | Gaussian with conjugate prior on mean. |
| [InputStream](#inputstream-1044) | Input stream. |
| [LogGaussian](#loggaussian-473) | Log-Gaussian distribution. |
| [LogGaussianExpression](#loggaussianexpression-947) | Expression that is a scaling of a log-Gaussian variable. |
| [LogGaussianWithGaussianExpressionMean](#loggaussianwithgaussianexpressionmean-655) | Log-Gaussian with conjugate prior on mean. |
| [LogGaussianWithGaussianLogExpressionMean](#loggaussianwithgaussianlogexpressionmean-674) | Log-Gaussian with conjugate prior on mean. |
| [LogGaussianWithGaussianMean](#loggaussianwithgaussianmean-693) | Log-Gaussian with conjugate prior on mean. |
| [MultivariateGaussian](#multivariategaussian-499) | Multivariate Gaussian distribution. |
| [MultivariateGaussianExpression](#multivariategaussianexpression-980) | Expression that is an affine transformation of a multivariate Gaussian. |
| [MultivariateGaussianWithMultivariateGaussianExpressionMean](#multivariategaussianwithmultivariategaussianexpressionmean-716) | Multivariate Gaussian with conjugate prior on mean. |
| [MultivariateGaussianWithMultivariateGaussianMean](#multivariategaussianwithmultivariategaussianmean-736) | Multivariate Gaussian with conjugate prior on mean. |
| [NegativeBinomial](#negativebinomial-521) | Negative binomial distribution. |
| [OutputStream](#outputstream-1075) | Output stream. |
| [Poisson](#poisson-535) | Poisson distribution. |
| [StdErrStream](#stderrstream-1076) | Output stream for stderr. |
| [StdInStream](#stdinstream-1079) | Input stream for stdin. |
| [StdOutStream](#stdoutstream-1082) | Output stream for stdout. |
| [Uniform](#uniform-551) | Uniform distribution. |


# Function Details

#### Bernoulli(ρ:[Real](#real-145)) -> [Bernoulli](#bernoulli-377)

<a name="bernoulli-380"></a>

Create Bernoulli distribution.

#### Beta(α:[Real](#real-145), β:[Real](#real-145)) -> [Beta](#beta-393)

<a name="beta-397"></a>

Create Beta distribution.

#### Binomial(n:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Binomial](#binomial-410)

<a name="binomial-414"></a>

Create binomial distribution.

#### Binomial(n:[Integer](#integer-22), ρ:[Beta](#beta-393)) -> [BetaBinomial](#betabinomial-567)

<a name="binomial-571"></a>

Create beta-binomial distribution.

#### Boolean(x:[Boolean](#boolean-1)) -> [Boolean](#boolean-1)

<a name="boolean-3"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-145), θ:[Real](#real-145)) -> [Gamma](#gamma-427)

<a name="gamma-431"></a>

Create Gamma distribution.

#### Gaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Gaussian](#gaussian-448)

<a name="gaussian-453"></a>

Create Gaussian distribution.

#### Gaussian(μ:[Real](#real-145)\[\_\], Σ:[Real](#real-145)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-499)

<a name="gaussian-505"></a>

Create multivariate Gaussian distribution.

#### I(rows:[Integer](#integer-22), columns:[Integer](#integer-22)) -> [Real](#real-145)\[\_,\_\]

<a name="i-1200"></a>

Create identity matrix.

#### Integer(x:[Integer64](#integer64-89)) -> [Integer](#integer-22)

<a name="integer-24"></a>

Convert other basic types to Integer. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer32(x:[Integer32](#integer32-33)) -> [Integer32](#integer32-33)

<a name="integer32-35"></a>

Convert other basic types to Integer32. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer64(x:[Integer64](#integer64-89)) -> [Integer64](#integer64-89)

<a name="integer64-91"></a>

Convert other basic types to Integer64. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### LogGaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [LogGaussian](#loggaussian-473)

<a name="loggaussian-478"></a>

Create log-Gaussian distribution.

#### LogNormal(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [LogGaussian](#loggaussian-473)

<a name="lognormal-481"></a>

Create log-Gaussian distribution.

#### NegativeBinomial(k:[Integer](#integer-22), ρ:[Real](#real-145)) -> [NegativeBinomial](#negativebinomial-521)

<a name="negativebinomial-525"></a>

Create.

#### Normal(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Gaussian](#gaussian-448)

<a name="normal-456"></a>

Create Gaussian distribution.

#### Normal(μ:[Real](#real-145)\[\_\], Σ:[Real](#real-145)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-499)

<a name="normal-508"></a>

Create multivariate Gaussian distribution.

#### Poisson(λ:[Real](#real-145)) -> [Poisson](#poisson-535)

<a name="poisson-538"></a>

Create Poisson distribution.

#### Poisson(λ:[Gamma](#gamma-427)) -> [GammaPoisson](#gammapoisson-583)

<a name="poisson-586"></a>

Create gamma-Poisson distribution.

#### Real(x:[Real64](#real64-214)) -> [Real](#real-145)

<a name="real-147"></a>

Convert other basic types to Real. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real32(x:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="real32-158"></a>

Convert other basic types to Real32. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### Real64(x:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="real64-216"></a>

Convert other basic types to Real64. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### String(x:[String](#string-272)) -> [String](#string-272)

<a name="string-274"></a>

Convert other basic types to String. This is overloaded for Bolean, Real64,
String, Integer64, Integer32 and String.

#### Uniform(l:[Real](#real-145), u:[Real](#real-145)) -> [Uniform](#uniform-551)

<a name="uniform-555"></a>

Create a Uniform distribution.

#### abs(x:[Integer32](#integer32-33)) -> [Integer32](#integer32-33)

<a name="abs-79"></a>

Absolute value.

#### abs(x:[Integer64](#integer64-89)) -> [Integer64](#integer64-89)

<a name="abs-135"></a>

Absolute value.

#### abs(x:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="abs-202"></a>

Absolute value.

#### abs(x:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="abs-260"></a>

Absolute value.

#### adjacent_difference(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="adjacent-difference-1468"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-145)\[\_\]) -> [Integer](#integer-22)

<a name="ancestor-1479"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-145)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="ancestors-1473"></a>

Sample an ancestry vector for a vector of log-weights.

#### beta(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="beta-1283"></a>

The beta function.

#### beta(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="beta-1286"></a>

The beta function.

#### choose(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="choose-1295"></a>

The binomial coefficient.

#### choose(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="choose-1298"></a>

The binomial coefficient.

#### columns(X:[Real](#real-145)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="columns-1163"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-22)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="columns-1165"></a>

Number of columns of a matrix.

#### columns(X:[Boolean](#boolean-1)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="columns-1167"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-22)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="cumulative-offspring-to-ancestors-1494"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="cumulative-weights-1506"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)

<a name="determinant-1140"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="exclusive-prefix-sum-1464"></a>

Inclusive prefix sum.

#### fclose(file:[File](#file-14))

<a name="fclose-21"></a>

Close a file.

#### fopen(file:[String](#string-272)) -> [File](#file-14)

<a name="fopen-16"></a>

Open a file for reading.

  - file : The file name.

#### fopen(file:[String](#string-272), mode:[String](#string-272)) -> [File](#file-14)

<a name="fopen-19"></a>

Open a file.

  - file : The file name.
  - mode : The mode, either `r` (read), `w` (write), `a` (append) or any
    other modes as in system `fopen`.

#### gamma(x:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="gamma-1274"></a>

The gamma function.

#### gamma(x:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="gamma-1276"></a>

The gamma function.

#### inclusive_prefix_sum(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="inclusive-prefix-sum-1460"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="inverse-1144"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-156)) -> [Boolean](#boolean-1)

<a name="isnan-213"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-214)) -> [Boolean](#boolean-1)

<a name="isnan-271"></a>

Does this have the value NaN?

#### lbeta(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="lbeta-1289"></a>

Logarithm of the beta function.

#### lbeta(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="lbeta-1292"></a>

Logarithm of the beta function.

#### lchoose(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="lchoose-1301"></a>

Logarithm of the binomial coefficient.

#### lchoose(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="lchoose-1304"></a>

Logarithm of the binomial coefficient.

#### length(x:[String](#string-272)) -> [Integer](#integer-22)

<a name="length-337"></a>

Length of a string.

#### length(x:[Real](#real-145)\[\_\]) -> [Integer64](#integer64-89)

<a name="length-1414"></a>

Length of a vector.

#### length(x:[Integer](#integer-22)\[\_\]) -> [Integer64](#integer64-89)

<a name="length-1416"></a>

Length of a vector.

#### length(x:[Boolean](#boolean-1)\[\_\]) -> [Integer64](#integer64-89)

<a name="length-1418"></a>

Length of a vector.

#### lgamma(x:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="lgamma-1278"></a>

Logarithm of the gamma function.

#### lgamma(x:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="lgamma-1280"></a>

Logarithm of the gamma function.

#### llt(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="llt-1147"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="log-sum-exp-1456"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-145), rows:[Integer](#integer-22), columns:[Integer](#integer-22)) -> [Real](#real-145)\[\_,\_\]

<a name="matrix-1174"></a>

Create matrix filled with a given scalar value.

#### matrix(x:[Integer](#integer-22), rows:[Integer](#integer-22), columns:[Integer](#integer-22)) -> [Integer](#integer-22)\[\_,\_\]

<a name="matrix-1181"></a>

Create matrix filled with a given scalar value.

#### matrix(x:[Boolean](#boolean-1), rows:[Integer](#integer-22), columns:[Integer](#integer-22)) -> [Boolean](#boolean-1)\[\_,\_\]

<a name="matrix-1188"></a>

Create matrix filled with a given scalar value.

#### max(x:[Integer32](#integer32-33), y:[Integer32](#integer32-33)) -> [Integer32](#integer32-33)

<a name="max-85"></a>

Maximum of two values.

#### max(x:[Integer64](#integer64-89), y:[Integer64](#integer64-89)) -> [Integer64](#integer64-89)

<a name="max-141"></a>

Maximum of two values.

#### max(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="max-208"></a>

Maximum of two values.

#### max(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="max-266"></a>

Maximum of two values.

#### max(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="max-1447"></a>

Maximum of a vector.

#### min(x:[Integer32](#integer32-33), y:[Integer32](#integer32-33)) -> [Integer32](#integer32-33)

<a name="min-88"></a>

Minimum of two values.

#### min(x:[Integer64](#integer64-89), y:[Integer64](#integer64-89)) -> [Integer64](#integer64-89)

<a name="min-144"></a>

Minimum of two values.

#### min(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="min-211"></a>

Minimum of two values.

#### min(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="min-269"></a>

Minimum of two values.

#### min(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="min-1451"></a>

Minimum of a vector.

#### mod(x:[Integer32](#integer32-33), y:[Integer32](#integer32-33)) -> [Integer32](#integer32-33)

<a name="mod-82"></a>

Modulus.

#### mod(x:[Integer64](#integer64-89), y:[Integer64](#integer64-89)) -> [Integer64](#integer64-89)

<a name="mod-138"></a>

Modulus.

#### mod(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="mod-205"></a>

Modulus.

#### mod(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="mod-263"></a>

Modulus.

#### norm(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="norm-1136"></a>

Norm of a vector.

#### observe_bernoulli(x:[Boolean](#boolean-1), ρ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-bernoulli-1203"></a>

Observe a Bernoulli variate.

- x: The variate.
- ρ: Probability of a true result.

Returns the log probability mass.

#### observe_beta(x:[Real](#real-145), α:[Real](#real-145), β:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-beta-1239"></a>

Observe a beta variate.

- x: The variate.
- α: Shape.
- β: Shape.

Returns the log probability density.

#### observe_beta_binomial(x:[Integer](#integer-22), n:[Integer](#integer-22), α:[Real](#real-145), β:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-beta-binomial-1216"></a>

Observe a beta-binomial variate.

- x: The variate.
- n: Number of trials.
- α: Shape.
- β: Shape.

Returns the log probability density.

Returns the log probability mass.

#### observe_binomial(x:[Integer](#integer-22), n:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-binomial-1207"></a>

Observe a binomial variate.

- x: The variate.
- n: Number of trials.
- ρ: Probability of a true result.

Returns the log probability mass.

#### observe_gamma(x:[Real](#real-145), k:[Real](#real-145), θ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-gamma-1235"></a>

Observe a gamma variate.

- x: The variate.
- k: Shape.
- θ: Scale.

Returns the log probability density.

#### observe_gaussian(x:[Real](#real-145), μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-gaussian-1227"></a>

Observe a Gaussian variate.

- x: The variate.
- μ: Mean.
- σ2: Variance.

Returns the log probability density.

#### observe_log_gaussian(x:[Real](#real-145), μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-log-gaussian-1231"></a>

Observe a log-Gaussian variate.

- x: The variate.
- μ: Mean.
- σ2: Variance.

Returns the log probability density.

#### observe_negative_binomial(x:[Integer](#integer-22), k:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-negative-binomial-1211"></a>

Observe a negative binomial variate.

- x: The variate (number of failures).
- k: Number of successes before the experiment is stopped.
- ρ: Probability of success.

Returns the log probability mass.

#### observe_poisson(x:[Integer](#integer-22), λ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-poisson-1219"></a>

Observe a Poisson variate.

- x: The variate.
- λ: Rate.

Returns the log probability mass.

#### observe_uniform(x:[Real](#real-145), l:[Real](#real-145), u:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-uniform-1223"></a>

Observe a uniform variate.

- x: The variate.
- l: Lower bound of interval.
- u: Upper bound of interval.

Returns the log probability density.

#### permute_ancestors(a:[Integer](#integer-22)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="permute-ancestors-1500"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### rows(X:[Real](#real-145)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="rows-1157"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-22)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="rows-1159"></a>

Number of rows of a matrix.

#### rows(X:[Boolean](#boolean-1)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="rows-1161"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)

<a name="scalar-1190"></a>

Convert single-element matrix to scalar value.

#### scalar(X:[Integer](#integer-22)\[\_,\_\]) -> [Integer](#integer-22)

<a name="scalar-1192"></a>

Convert single-element matrix to scalar value.

#### scalar(X:[Boolean](#boolean-1)\[\_,\_\]) -> [Boolean](#boolean-1)

<a name="scalar-1194"></a>

Convert single-element matrix to scalar value.

#### scalar(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="scalar-1435"></a>

Convert single-element vector to scalar value.

#### scalar(x:[Integer](#integer-22)\[\_\]) -> [Integer](#integer-22)

<a name="scalar-1437"></a>

Convert single-element vector to scalar value.

#### scalar(x:[Boolean](#boolean-1)\[\_\]) -> [Boolean](#boolean-1)

<a name="scalar-1439"></a>

Convert single-element vector to scalar value.

#### seed(s:[Integer](#integer-22))

<a name="seed-1241"></a>

Seed the pseudorandom number generator.

- seed: Seed value.

#### simulate_bernoulli(ρ:[Real](#real-145)) -> [Boolean](#boolean-1)

<a name="simulate-bernoulli-1243"></a>

Simulate a Bernoulli variate.

- ρ: Probability of a true result.

#### simulate_beta(α:[Real](#real-145), β:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-beta-1272"></a>

Simulate a beta variate.

- α: Shape.
- β: Shape.

#### simulate_beta_binomial(n:[Integer](#integer-22), α:[Real](#real-145), β:[Real](#real-145)) -> [Integer](#integer-22)

<a name="simulate-beta-binomial-1253"></a>

Simulate a beta-binomial variate.

- n: Number of trials.
- α: Shape.
- β: Shape.

#### simulate_binomial(n:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Integer](#integer-22)

<a name="simulate-binomial-1246"></a>

Simulate a binomial variate.

- n: Number of trials.
- ρ: Probability of a true result.

#### simulate_gamma(k:[Real](#real-145), θ:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-gamma-1267"></a>

Simulate a gamma variate.

- k: Shape.
- θ: Scale.

#### simulate_gaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-gaussian-1261"></a>

Simulate a Gaussian variate.

- μ: Mean.
- σ2: Variance.

#### simulate_log_gaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-log-gaussian-1264"></a>

Simulate a log-Gaussian variate.

- μ: Mean (in log space).
- σ2: Variance (in log space).

#### simulate_negative_binomial(k:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Integer](#integer-22)

<a name="simulate-negative-binomial-1249"></a>

Simulate a negative binomial variate.

- k: Number of successes before the experiment is stopped.
- ρ: Probability of success.

Returns the number of failures.

#### simulate_poisson(λ:[Real](#real-145)) -> [Integer](#integer-22)

<a name="simulate-poisson-1255"></a>

Simulate a Poisson variate.

- λ: Rate.

#### simulate_uniform(l:[Real](#real-145), u:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-uniform-1258"></a>

Simulate a uniform variate.

- l: Lower bound of interval.
- u: Upper bound of interval.

#### solve(X:[Real](#real-145)\[\_,\_\], y:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="solve-1151"></a>

Solve a system of equations.

#### solve(X:[Real](#real-145)\[\_,\_\], Y:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="solve-1155"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="squarednorm-1138"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="sum-1443"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-145)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="systematic-cumulative-offspring-1486"></a>

Systematic resampling.

#### transpose(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="transpose-1142"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-145), length:[Integer](#integer-22)) -> [Real](#real-145)\[\_\]

<a name="vector-1423"></a>

Create vector filled with a given scalar.

#### vector(x:[Integer](#integer-22), length:[Integer](#integer-22)) -> [Integer](#integer-22)\[\_\]

<a name="vector-1428"></a>

Create vector filled with a given value.

#### vector(x:[Boolean](#boolean-1), length:[Integer](#integer-22)) -> [Boolean](#boolean-1)\[\_\]

<a name="vector-1433"></a>

Create vector filled with a given value.


# Program Details

#### build(include_dir:[String](#string-272), lib_dir:[String](#string-272), share_dir:[String](#string-272), prefix:[String](#string-272), warnings:[Boolean](#boolean-1) <- true, debug:[Boolean](#boolean-1) <- true, verbose:[Boolean](#boolean-1) <- true)

<a name="build-1026"></a>

Build the project.

  - `--include-dir` : Add search directory for header files.
  - `--lib-dir` : Add search directory for library files.
  - `--share-dir` : Add search directory for data files.

These three options are analogous to their counterparts for a C/C++
compiler, and specify the locations in which the Birch compiler should
search for headers (both Birch and C/C++ headers), installed libraries and
installed data files. They may be given multiple times to specify multiple
directories in the order in which they are to be searched.

After searching these directories, the Birch compiler will search the
environment variables `BIRCH_INCLUDE_PATH`, `BIRCH_LIBRARY_PATH` and
`BIRCH_SHARE_PATH`, followed by the directories of the compiler's own
installation, followed by the system-wide locations `/usr/local/` and
`/usr/`.

  - `--prefix` : Installation prefix (default platform-specific).
  - `--enable-std` / `--disable-std` : Enable/disable the standard library.
  - `--enable-warnings` / `--disable-warnings` : Enable/disable warnings.
  - `--enable-debug` / `--disable-debug` : Enable/disable debug mode.
  - `--enable-verbose` / `--disable-verbose` : Verbose mode.

#### check()

<a name="check-1027"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-1028"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-1029"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-1030"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-272) <- "untitled")

<a name="init-1032"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-1033"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-1034"></a>

Uninstall the project. This uninstalls all header, library and data files
from the directory specified by `--prefix` (or the system default if this
was not specified).


# Basic Type Details

#### type Boolean;

<a name="boolean-1"></a>

A Boolean value.

#### type File;

<a name="file-14"></a>

A file handle.

#### type Integer32 < [Integer64](#integer64-89);

<a name="integer32-33"></a>

A 32-bit integer.

#### type Integer64 < [Real64](#real64-214);

<a name="integer64-89"></a>

A 64-bit integer.

#### type Real32 < [Real64](#real64-214);

<a name="real32-156"></a>

A 32-bit (single precision) floating point value.

#### type Real64;

<a name="real64-214"></a>

A 64-bit (double precision) floating point value.

#### type String;

<a name="string-272"></a>

A string value.


# Alias Type Details

#### type Integer = [Integer64](#integer64-89);


<a name="integer-22"></a>

An integer value of default type.

#### type LogNormal = [LogGaussian](#loggaussian-473);


<a name="lognormal-474"></a>

Synonym for LogGaussian.

#### type MultivariateNormal = [MultivariateGaussian](#multivariategaussian-499);


<a name="multivariatenormal-500"></a>

Synonym for MultivariateGaussian.

#### type Normal = [Gaussian](#gaussian-448);


<a name="normal-449"></a>

Synonym for Gaussian.

#### type Real = [Real64](#real64-214);


<a name="real-145"></a>

A floating point value of default type.


# Class Type Details


## Bernoulli

<a name="bernoulli-377"></a>

  * Inherits from *[DelayBoolean](#delayboolean-800)*

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-145)* | Probability of a true result. |


## Beta

<a name="beta-393"></a>

  * Inherits from *[DelayReal](#delayreal-824)*

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-145)* | First shape parameter. |
| *β:[Real](#real-145)* | Second shape parameter. |


## BetaBinomial

<a name="betabinomial-567"></a>

  * Inherits from *[DelayInteger](#delayinteger-812)*

Binomial with conjugate prior on success probability. When the success
probability is known, this is simply a Binomial distribution. When unknown,

| Member Variable | Description |
| --- | --- |
| *n:[Integer](#integer-22)* | Number of trials. |
| *ρ:[Beta](#beta-393)* | Probability of a true result. |


## Binomial

<a name="binomial-410"></a>

  * Inherits from *[DelayInteger](#delayinteger-812)*

Binomial distribution.

| Member Variable | Description |
| --- | --- |
| *n:[Integer](#integer-22)* | Number of trials. |
| *ρ:[Real](#real-145)* | Probability of a true result. |


## Delay

<a name="delay-788"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-22)* | State of the variate. |
| *missing:[Boolean](#boolean-1)* | Is the value missing? |
| *parent:[Delay](#delay-788)?* | Parent. |
| *child:[Delay](#delay-788)?* | Child, if one exists and it is on the stem. |
| *id:[Integer](#integer-22)* | Unique id for delayed sampling diagnostics. |
| *nforward:[Integer](#integer-22)* | Number of observations absorbed in forward pass, for delayed sampling diagnostics. |
| *nbackward:[Integer](#integer-22)* | Number of observations absorbed in backward pass, for delayed sampling diagnostics. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-755) | Is this a root node? |
| [isTerminal](#isterminal-756) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-757) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-758) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-759) | Is this node in the marginalized state? |
| [isRealized](#isrealized-760) | Is this node in the realized state? |
| [isMissing](#ismissing-761) | Is the value of this node missing? |
| [isNotMissing](#isnotmissing-762) | Is the value of this node not missing? |
| [initialize](#initialize-763) | Initialize as a root node. |
| [initialize](#initialize-765) | Initialize as a non-root node. |
| [absorb](#absorb-767) | Increment number of observations absorbed. |
| [marginalize](#marginalize-768) | Marginalize the variate. |
| [simulate](#simulate-769) | Simulate the variate. |
| [observe](#observe-770) | Observe the variate. |
| [realize](#realize-771) | Realize the variate. |
| [graft](#graft-772) | Graft the stem to this node. |
| [graft](#graft-774) | Graft the stem to this node. |
| [prune](#prune-775) | Prune the stem from below this node. |
| [setParent](#setparent-777) | Set the parent. |
| [removeParent](#removeparent-778) | Remove the parent. |
| [setChild](#setchild-780) | Set the child. |
| [removeChild](#removechild-781) | Remove the child. |
| [register](#register-786) | Register with the diagnostic handler. |
| [trigger](#trigger-787) | Trigger an event with the diagnostic handler. |


### Member Function Details

#### absorb(nbackward:[Integer](#integer-22))

<a name="absorb-767"></a>

Increment number of observations absorbed.

  - `nbackward` : Number of new observations absorbed.

#### graft()

<a name="graft-772"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-788))

<a name="graft-774"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-763"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-788))

<a name="initialize-765"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-1)

<a name="isinitialized-758"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-1)

<a name="ismarginalized-759"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-1)

<a name="ismissing-761"></a>

Is the value of this node missing?

#### isNotMissing() -> [Boolean](#boolean-1)

<a name="isnotmissing-762"></a>

Is the value of this node not missing?

#### isRealized() -> [Boolean](#boolean-1)

<a name="isrealized-760"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-1)

<a name="isroot-755"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-1)

<a name="isterminal-756"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-1)

<a name="isuninitialized-757"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-768"></a>

Marginalize the variate.

#### observe()

<a name="observe-770"></a>

Observe the variate.

#### prune()

<a name="prune-775"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-771"></a>

Realize the variate.

#### register()

<a name="register-786"></a>

Register with the diagnostic handler.

#### removeChild()

<a name="removechild-781"></a>

Remove the child.

#### removeParent()

<a name="removeparent-778"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-788))

<a name="setchild-780"></a>

Set the child.

#### setParent(u:[Delay](#delay-788))

<a name="setparent-777"></a>

Set the parent.

#### simulate()

<a name="simulate-769"></a>

Simulate the variate.

#### trigger()

<a name="trigger-787"></a>

Trigger an event with the diagnostic handler.


## DelayBoolean

<a name="delayboolean-800"></a>

  * Inherits from *[Delay](#delay-788)*

Abstract delay variate with Boolean value.

| Assignment | Description |
| --- | --- |
| *[Boolean](#boolean-1)* | Value assignment. |
| *[String](#string-272)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Boolean](#boolean-1)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Boolean](#boolean-1)* | Value. |
| *w:[Real](#real-145)* | Weight. |


## DelayDiagnostics

<a name="delaydiagnostics-367"></a>

Outputs graphical representations of the delayed sampling state for
diagnostic purposes.

  - N : Maximum number of nodes

To use, before running any code that uses delayed sampling, construct a
`DelayDiagnostic` object with sufficient capacity to hold all nodes.
Then, use `name()` to give a name to each node that will be of interest
to output, and optionally `position()` to give an explicit position.
Variates that are not named are not output.

See the Birch example programs, e.g. `delay_triplet` and `delay_kalman`
for an example of how this is done.

On each event, will output a `*.dot` file into `diagnostics/stateN.dot`,
where `N` is the event number. If positions have been explicitly given
using `position()`, it is recommended that these are compiled with
`neato`, otherwise, if positions have not been explicitly given so that
automatic layout is desired, use `dot`, e.g.

    dot -Tpdf diagnostics/state1.dot > diagnostics/state1.pdf

| Member Variable | Description |
| --- | --- |
| *nodes:[Delay](#delay-788)?\[\_\]* | Registered nodes. |
| *names:[String](#string-272)?\[\_\]* | Names of the nodes. |
| *xs:[Integer](#integer-22)?\[\_\]* | $x$-coordinates of the nodes. |
| *ys:[Integer](#integer-22)?\[\_\]* | $y$-coordinates of the nodes. |
| *n:[Integer](#integer-22)* | Number of nodes that have been registered. |
| *noutputs:[Integer](#integer-22)* | Number of graphs that have been output. |

| Member Function | Brief description |
| --- | --- |
| [register](#register-347) | Register a new node. |
| [name](#name-350) | Set the name of a node. |
| [position](#position-354) | Set the position of a node. |
| [trigger](#trigger-357) | Trigger an event. |
| [dot](#dot-366) | Output a dot graph of the current state. |


### Member Function Details

#### dot()

<a name="dot-366"></a>

Output a dot graph of the current state.

#### name(id:[Integer](#integer-22), name:[String](#string-272))

<a name="name-350"></a>

Set the name of a node.

  - id   : Id of the node.
  - name : The name.

#### position(id:[Integer](#integer-22), x:[Integer](#integer-22), y:[Integer](#integer-22))

<a name="position-354"></a>

Set the position of a node.

  - id : Id of the node.
  - x  : $x$-coordinate.
  - y  : $y$-coordinate.

#### register(o:[Delay](#delay-788)) -> [Integer](#integer-22)

<a name="register-347"></a>

Register a new node. This is a callback function typically called
within the Delay class itself.

Returns an id assigned to the node.

#### trigger()

<a name="trigger-357"></a>

Trigger an event.


## DelayInteger

<a name="delayinteger-812"></a>

  * Inherits from *[Delay](#delay-788)*

Abstract delay variate with Integer value.

| Assignment | Description |
| --- | --- |
| *[Integer](#integer-22)* | Value assignment. |
| *[String](#string-272)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Integer](#integer-22)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Integer](#integer-22)* | Value. |
| *w:[Real](#real-145)* | Weight. |


## DelayReal

<a name="delayreal-824"></a>

  * Inherits from *[Delay](#delay-788)*

Abstract delay variate with Real value.

| Assignment | Description |
| --- | --- |
| *[Real](#real-145)* | Value assignment. |
| *[String](#string-272)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-145)* | Value. |
| *w:[Real](#real-145)* | Weight. |


## DelayRealVector

<a name="delayrealvector-835"></a>

  * Inherits from *[Delay](#delay-788)*

Abstract delay variate with real vector value.

`D` Number of dimensions.

| Assignment | Description |
| --- | --- |
| *[Real](#real-145)\[\_\]* | Value assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-145)\[\_\]* | Value. |
| *w:[Real](#real-145)* | Weight. |


## FileInputStream

<a name="fileinputstream-1037"></a>

  * Inherits from *[InputStream](#inputstream-1044)*

File input stream.

| Member Function | Brief description |
| --- | --- |
| [close](#close-1036) | Close the file. |


### Member Function Details

#### close()

<a name="close-1036"></a>

Close the file.


## FileOutputStream

<a name="fileoutputstream-1040"></a>

  * Inherits from *[OutputStream](#outputstream-1075)*

File output stream.

| Member Function | Brief description |
| --- | --- |
| [close](#close-1039) | Close the file. |


### Member Function Details

#### close()

<a name="close-1039"></a>

Close the file.


## Gamma

<a name="gamma-427"></a>

  * Inherits from *[DelayReal](#delayreal-824)*

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-145)* | Shape. |
| *θ:[Real](#real-145)* | Scale. |


## GammaPoisson

<a name="gammapoisson-583"></a>

  * Inherits from *[DelayInteger](#delayinteger-812)*

Poisson with conjugate prior on rate. When the rate is known, this is
simply a Poisson distribution. When the rate is gamma distributed, this is
a negative binomial distribution.

| Member Variable | Description |
| --- | --- |
| *λ:[Gamma](#gamma-427)* | Rate. |


## Gaussian

<a name="gaussian-448"></a>

  * Inherits from *[DelayReal](#delayreal-824)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-145)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |


## GaussianExpression

<a name="gaussianexpression-843"></a>

Expression that is an affine transformation of a Gaussian variable.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-145)* | Scale. |
| *x:[Gaussian](#gaussian-448)* | Base distribution. |
| *c:[Real](#real-145)* | Offset. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-842) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-145), x:[Gaussian](#gaussian-448), c:[Real](#real-145))

<a name="initialize-842"></a>

Initialize.


## GaussianLogExpression

<a name="gaussianlogexpression-911"></a>

Expression that is an affine transformation of the logarithm of a
log-Gaussian variable.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-145)* | Scale. |
| *x:[LogGaussian](#loggaussian-473)* | Base distribution. |
| *c:[Real](#real-145)* | Offset. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-910) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-145), x:[LogGaussian](#loggaussian-473), c:[Real](#real-145))

<a name="initialize-910"></a>

Initialize.


## GaussianWithGaussianExpressionMean

<a name="gaussianwithgaussianexpressionmean-598"></a>

  * Inherits from *[Gaussian](#gaussian-448)*

Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianExpression](#gaussianexpression-843)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## GaussianWithGaussianLogExpressionMean

<a name="gaussianwithgaussianlogexpressionmean-617"></a>

  * Inherits from *[Gaussian](#gaussian-448)*

Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianLogExpression](#gaussianlogexpression-911)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## GaussianWithGaussianMean

<a name="gaussianwithgaussianmean-636"></a>

  * Inherits from *[Gaussian](#gaussian-448)*

Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[Gaussian](#gaussian-448)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## InputStream

<a name="inputstream-1044"></a>

Input stream.

| Member Function | Brief description |
| --- | --- |
| [readInteger](#readinteger-1042) | Read integer. |
| [readReal](#readreal-1043) | Read real. |


### Member Function Details

#### readInteger() -> [Integer](#integer-22)

<a name="readinteger-1042"></a>

Read integer.

#### readReal() -> [Real](#real-145)

<a name="readreal-1043"></a>

Read real.


## LogGaussian

<a name="loggaussian-473"></a>

  * Inherits from *[DelayReal](#delayreal-824)*

Log-Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-145)* | Mean after log transformation. |
| *σ2:[Real](#real-145)* | Variance after log transformation. |


## LogGaussianExpression

<a name="loggaussianexpression-947"></a>

Expression that is a scaling of a log-Gaussian variable.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-145)* | Scale. |
| *x:[LogGaussian](#loggaussian-473)* | Base distribution. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-946) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-145), x:[LogGaussian](#loggaussian-473))

<a name="initialize-946"></a>

Initialize.


## LogGaussianWithGaussianExpressionMean

<a name="loggaussianwithgaussianexpressionmean-655"></a>

  * Inherits from *[LogGaussian](#loggaussian-473)*

Log-Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianExpression](#gaussianexpression-843)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## LogGaussianWithGaussianLogExpressionMean

<a name="loggaussianwithgaussianlogexpressionmean-674"></a>

  * Inherits from *[LogGaussian](#loggaussian-473)*

Log-Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianLogExpression](#gaussianlogexpression-911)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## LogGaussianWithGaussianMean

<a name="loggaussianwithgaussianmean-693"></a>

  * Inherits from *[LogGaussian](#loggaussian-473)*

Log-Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[Gaussian](#gaussian-448)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## MultivariateGaussian

<a name="multivariategaussian-499"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-835)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-145)\[\_\]* | Mean. |
| *Σ:[Real](#real-145)\[\_,\_\]* | Covariance. |


## MultivariateGaussianExpression

<a name="multivariategaussianexpression-980"></a>

Expression that is an affine transformation of a multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-145)\[\_,\_\]* | Scale. |
| *x:[MultivariateGaussian](#multivariategaussian-499)* | Base distribution. |
| *c:[Real](#real-145)\[\_\]* | Offset. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-979) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-145)\[\_,\_\], x:[MultivariateGaussian](#multivariategaussian-499), c:[Real](#real-145)\[\_\])

<a name="initialize-979"></a>

Initialize.


## MultivariateGaussianWithMultivariateGaussianExpressionMean

<a name="multivariategaussianwithmultivariategaussianexpressionmean-716"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-499)*

Multivariate Gaussian with conjugate prior on mean.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[MultivariateGaussianExpression](#multivariategaussianexpression-980)?* | Mean. |
| *Σ:[Real](#real-145)\[\_,\_\]* | Covariance. |
| *μ_0:[Real](#real-145)\[\_\]* | Marginalized prior mean. |
| *Σ_0:[Real](#real-145)\[\_,\_\]* | Marginalized prior covariance. |


## MultivariateGaussianWithMultivariateGaussianMean

<a name="multivariategaussianwithmultivariategaussianmean-736"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-499)*

Multivariate Gaussian with conjugate prior on mean.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[MultivariateGaussian](#multivariategaussian-499)* | Mean. |
| *Σ:[Real](#real-145)\[\_,\_\]* | Covariance. |
| *μ_0:[Real](#real-145)\[\_\]* | Marginalized prior mean. |
| *Σ_0:[Real](#real-145)\[\_,\_\]* | Marginalized prior covariance. |


## NegativeBinomial

<a name="negativebinomial-521"></a>

  * Inherits from *[DelayInteger](#delayinteger-812)*

Negative binomial distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Integer](#integer-22)* | Number of successes before the experiment is stopped. |
| *ρ:[Real](#real-145)* | Probability of success. |


## OutputStream

<a name="outputstream-1075"></a>

Output stream.

| Member Function | Brief description |
| --- | --- |
| [print](#print-1047) | Print string. |
| [print](#print-1049) | Print value. |
| [print](#print-1051) | Print value. |
| [print](#print-1053) | Print value. |
| [print](#print-1056) | Print vector. |
| [print](#print-1059) | Print vector. |
| [print](#print-1062) | Print vector. |
| [print](#print-1066) | Print matrix. |
| [print](#print-1070) | Print matrix. |
| [print](#print-1074) | Print matrix. |


### Member Function Details

#### print(value:[String](#string-272))

<a name="print-1047"></a>

Print string.

#### print(value:[Boolean](#boolean-1))

<a name="print-1049"></a>

Print value.

#### print(value:[Integer](#integer-22))

<a name="print-1051"></a>

Print value.

#### print(value:[Real](#real-145))

<a name="print-1053"></a>

Print value.

#### print(x:[Boolean](#boolean-1)\[\_\])

<a name="print-1056"></a>

Print vector.

#### print(x:[Integer](#integer-22)\[\_\])

<a name="print-1059"></a>

Print vector.

#### print(x:[Real](#real-145)\[\_\])

<a name="print-1062"></a>

Print vector.

#### print(X:[Boolean](#boolean-1)\[\_,\_\])

<a name="print-1066"></a>

Print matrix.

#### print(X:[Integer](#integer-22)\[\_,\_\])

<a name="print-1070"></a>

Print matrix.

#### print(X:[Real](#real-145)\[\_,\_\])

<a name="print-1074"></a>

Print matrix.


## Poisson

<a name="poisson-535"></a>

  * Inherits from *[DelayInteger](#delayinteger-812)*

Poisson distribution.

| Member Variable | Description |
| --- | --- |
| *λ:[Real](#real-145)* | Rate. |


## StdErrStream

<a name="stderrstream-1076"></a>

  * Inherits from *[OutputStream](#outputstream-1075)*

Output stream for stderr.


## StdInStream

<a name="stdinstream-1079"></a>

  * Inherits from *[InputStream](#inputstream-1044)*

Input stream for stdin.


## StdOutStream

<a name="stdoutstream-1082"></a>

  * Inherits from *[OutputStream](#outputstream-1075)*

Output stream for stdout.


## Uniform

<a name="uniform-551"></a>

  * Inherits from *[DelayReal](#delayreal-824)*

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-145)* | Lower bound. |
| *u:[Real](#real-145)* | Upper bound. |

