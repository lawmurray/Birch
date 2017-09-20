
# Summary

| Variable | Description |
| --- | --- |
| *delayDiagnostics:[DelayDiagnostics](#delaydiagnostics-367)?* | Global diagnostics handler for delayed sampling. |
| *inf:[Real64](#real64-214)* | $\infty$ |
| *stderr:[StdErrStream](#stderrstream-1009)* | Standard error. |
| *stdout:[StdOutStream](#stdoutstream-1012)* | Standard output. |
| *π:[Real64](#real64-214)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-817) | Create Bernoulli distribution. |
| [Beta](#beta-804) | Create Beta distribution. |
| [Binomial](#binomial-834) | Create binomial distribution. |
| [Boolean](#boolean-3) | Convert other basic types to Boolean. |
| [Gamma](#gamma-851) | Create Gamma distribution. |
| [Gaussian](#gaussian-873) | Create Gaussian distribution. |
| [Gaussian](#gaussian-919) | Create multivariate Gaussian distribution. |
| [Integer](#integer-24) | Convert other basic types to Integer. |
| [Integer32](#integer32-35) | Convert other basic types to Integer32. |
| [Integer64](#integer64-91) | Convert other basic types to Integer64. |
| [LogGaussian](#loggaussian-895) | Create log-Gaussian distribution. |
| [NegativeBinomial](#negativebinomial-936) | Create. |
| [Real](#real-147) | Convert other basic types to Real. |
| [Real32](#real32-158) | Convert other basic types to Real32. |
| [Real64](#real64-216) | Convert other basic types to Real64. |
| [String](#string-274) | Convert other basic types to String. |
| [Uniform](#uniform-953) | Create a Uniform distribution. |
| [abs](#abs-79) | Absolute value. |
| [abs](#abs-135) | Absolute value. |
| [abs](#abs-202) | Absolute value. |
| [abs](#abs-260) | Absolute value. |
| [adjacent_difference](#adjacent-difference-1346) | Inclusive prefix sum. |
| [ancestor](#ancestor-1363) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-1357) | Sample an ancestry vector for a vector of log-weights. |
| [beta](#beta-1177) | The beta function. |
| [beta](#beta-1180) | The beta function. |
| [choose](#choose-1189) | The binomial coefficient. |
| [choose](#choose-1192) | The binomial coefficient. |
| [columns](#columns-1091) | Number of columns of a matrix. |
| [columns](#columns-1093) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-1378) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-1390) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-1070) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-1342) | Inclusive prefix sum. |
| [fclose](#fclose-21) | Close a file. |
| [fopen](#fopen-16) | Open a file for reading. |
| [fopen](#fopen-19) | Open a file. |
| [gamma](#gamma-1168) | The gamma function. |
| [gamma](#gamma-1170) | The gamma function. |
| [identity](#identity-1108) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-1338) | Inclusive prefix sum. |
| [inverse](#inverse-1074) | Inverse of a matrix. |
| [isnan](#isnan-213) | Does this have the value NaN? |
| [isnan](#isnan-271) | Does this have the value NaN? |
| [lbeta](#lbeta-1183) | Logarithm of the beta function. |
| [lbeta](#lbeta-1186) | Logarithm of the beta function. |
| [lchoose](#lchoose-1195) | Logarithm of the binomial coefficient. |
| [lchoose](#lchoose-1198) | Logarithm of the binomial coefficient. |
| [length](#length-337) | Length of a string. |
| [length](#length-1308) | Length of a vector. |
| [length](#length-1310) | Length of a vector. |
| [lgamma](#lgamma-1172) | Logarithm of the gamma function. |
| [lgamma](#lgamma-1174) | Logarithm of the gamma function. |
| [llt](#llt-1077) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-1334) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-1102) | Create matrix filled with a given scalar. |
| [max](#max-85) | Maximum of two values. |
| [max](#max-141) | Maximum of two values. |
| [max](#max-208) | Maximum of two values. |
| [max](#max-266) | Maximum of two values. |
| [max](#max-1325) | Maximum of a vector. |
| [min](#min-88) | Minimum of two values. |
| [min](#min-144) | Minimum of two values. |
| [min](#min-211) | Minimum of two values. |
| [min](#min-269) | Minimum of two values. |
| [min](#min-1329) | Minimum of a vector. |
| [mod](#mod-82) | Modulus. |
| [mod](#mod-138) | Modulus. |
| [mod](#mod-205) | Modulus. |
| [mod](#mod-263) | Modulus. |
| [norm](#norm-1066) | Norm of a vector. |
| [observe_bernoulli](#observe-bernoulli-1111) | Observe a Bernoulli variate. |
| [observe_beta](#observe-beta-1139) | Observe a Beta variate. |
| [observe_binomial](#observe-binomial-1115) | Observe a Binomial variate. |
| [observe_gamma](#observe-gamma-1135) | Observe a Gamma variate. |
| [observe_gaussian](#observe-gaussian-1127) | Observe a Gaussian variate. |
| [observe_log_gaussian](#observe-log-gaussian-1131) | Observe a log-Gaussian variate. |
| [observe_negative_binomial](#observe-negative-binomial-1119) | Observe a Negative Binomial variate. |
| [observe_uniform](#observe-uniform-1123) | Observe a Uniform variate. |
| [permute_ancestors](#permute-ancestors-1384) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [read](#read-1352) | Read numbers from a file. |
| [rows](#rows-1087) | Number of rows of a matrix. |
| [rows](#rows-1089) | Number of rows of a matrix. |
| [scalar](#scalar-1095) | Convert single-element matrix to scalar. |
| [scalar](#scalar-1312) | Convert single-element vector to scalar. |
| [seed](#seed-1141) | Seed the pseudorandom number generator. |
| [simulate_bernoulli](#simulate-bernoulli-1143) | Simulate a Bernoulli variate. |
| [simulate_beta](#simulate-beta-1166) | Simulate a Beta variate. |
| [simulate_binomial](#simulate-binomial-1146) | Simulate a Binomial variate. |
| [simulate_gamma](#simulate-gamma-1161) | Simulate a Gamma variate. |
| [simulate_gaussian](#simulate-gaussian-1155) | Simulate a Gaussian variate. |
| [simulate_log_gaussian](#simulate-log-gaussian-1158) | Simulate a log-Gaussian variate. |
| [simulate_negative_binomial](#simulate-negative-binomial-1149) | Simulate a Negative Binomial variate. |
| [simulate_uniform](#simulate-uniform-1152) | Simulate a Uniform variate. |
| [solve](#solve-1081) | Solve a system of equations. |
| [solve](#solve-1085) | Solve a system of equations. |
| [squaredNorm](#squarednorm-1068) | Squared norm of a vector. |
| [sum](#sum-1321) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-1370) | Systematic resampling. |
| [transpose](#transpose-1072) | Transpose of a matrix. |
| [vector](#vector-1317) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-961) | Build the project. |
| [check](#check-962) | Check the file structure of the project for possible issues. |
| [clean](#clean-963) | Clean the project directory of all build files. |
| [dist](#dist-964) | Build a distributable archive for the project. |
| [docs](#docs-965) | Build the reference documentation for the project. |
| [init](#init-967) | Initialise the working directory for a new project. |
| [install](#install-968) | Install the project. |
| [uninstall](#uninstall-969) | Uninstall the project. |

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
| [LogNormal](#lognormal-891) | Synonym for LogGaussian. |
| [MultivariateNormal](#multivariatenormal-914) | Synonym for MultivariateGaussian. |
| [Normal](#normal-869) | Synonym for Gaussian. |
| [Real](#real-145) | A floating point value of default type. |

| Class Type | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-814) | Bernoulli distribution. |
| [Beta](#beta-800) | Beta distribution. |
| [Binomial](#binomial-830) | Binomial distribution. |
| [Delay](#delay-545) | Node interface for delayed sampling. |
| [DelayBoolean](#delayboolean-560) | Abstract delay variate with Boolean value. |
| [DelayDiagnostics](#delaydiagnostics-367) | Outputs graphical representations of the delayed sampling state for diagnostic purposes. |
| [DelayInteger](#delayinteger-575) | Abstract delay variate with Integer value. |
| [DelayReal](#delayreal-590) | Abstract delay variate with Real value. |
| [DelayRealVector](#delayrealvector-604) | Abstract delay variate with real vector value. |
| [FileOutputStream](#fileoutputstream-972) | File output stream. |
| [Gamma](#gamma-847) | Gamma distribution. |
| [Gaussian](#gaussian-868) | Gaussian distribution. |
| [GaussianExpression](#gaussianexpression-612) | Expression that is an affine transformation of a Gaussian variable. |
| [GaussianLogExpression](#gaussianlogexpression-680) | Expression that is an affine transformation of the logarithm of a log-Gaussian variable. |
| [GaussianWithGaussianExpressionMean](#gaussianwithgaussianexpressionmean-379) | Gaussian with conjugate prior on mean. |
| [GaussianWithGaussianLogExpressionMean](#gaussianwithgaussianlogexpressionmean-395) | Gaussian with conjugate prior on mean. |
| [GaussianWithGaussianMean](#gaussianwithgaussianmean-411) | Gaussian with conjugate prior on mean. |
| [LogGaussian](#loggaussian-890) | Log-Gaussian distribution. |
| [LogGaussianExpression](#loggaussianexpression-716) | Expression that is a scaling of a log-Gaussian variable. |
| [LogGaussianWithGaussianExpressionMean](#loggaussianwithgaussianexpressionmean-427) | Log-Gaussian with conjugate prior on mean. |
| [LogGaussianWithGaussianLogExpressionMean](#loggaussianwithgaussianlogexpressionmean-443) | Log-Gaussian with conjugate prior on mean. |
| [LogGaussianWithGaussianMean](#loggaussianwithgaussianmean-459) | Log-Gaussian with conjugate prior on mean. |
| [MultivariateGaussian](#multivariategaussian-913) | Multivariate Gaussian distribution. |
| [MultivariateGaussianExpression](#multivariategaussianexpression-749) | Expression that is an affine transformation of a multivariate Gaussian. |
| [MultivariateGaussianWithMultivariateGaussianExpressionMean](#multivariategaussianwithmultivariategaussianexpressionmean-479) | Multivariate Gaussian with conjugate prior on mean. |
| [MultivariateGaussianWithMultivariateGaussianMean](#multivariategaussianwithmultivariategaussianmean-496) | Multivariate Gaussian with conjugate prior on mean. |
| [NegativeBinomial](#negativebinomial-932) | Negative binomial distribution. |
| [OutputStream](#outputstream-1008) | Output stream. |
| [StdErrStream](#stderrstream-1009) | Output stream for stderr. |
| [StdOutStream](#stdoutstream-1012) | Output stream for stdout. |
| [Uniform](#uniform-949) | Uniform distribution. |


# Function Details

#### Bernoulli(ρ:[Real](#real-145)) -> [Bernoulli](#bernoulli-814)

<a name="bernoulli-817"></a>

Create Bernoulli distribution.

#### Beta(α:[Real](#real-145), β:[Real](#real-145)) -> [Beta](#beta-800)

<a name="beta-804"></a>

Create Beta distribution.

#### Binomial(n:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Binomial](#binomial-830)

<a name="binomial-834"></a>

Create binomial distribution.

#### Boolean(x:[Boolean](#boolean-1)) -> [Boolean](#boolean-1)

<a name="boolean-3"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-145), θ:[Real](#real-145)) -> [Gamma](#gamma-847)

<a name="gamma-851"></a>

Create Gamma distribution.

#### Gaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Gaussian](#gaussian-868)

<a name="gaussian-873"></a>

Create Gaussian distribution.

#### Gaussian(μ:[Real](#real-145)\[\_\], Σ:[Real](#real-145)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-913)

<a name="gaussian-919"></a>

Create multivariate Gaussian distribution.

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

#### LogGaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [LogGaussian](#loggaussian-890)

<a name="loggaussian-895"></a>

Create log-Gaussian distribution.

#### NegativeBinomial(k:[Integer](#integer-22), ρ:[Real](#real-145)) -> [NegativeBinomial](#negativebinomial-932)

<a name="negativebinomial-936"></a>

Create.

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

#### Uniform(l:[Real](#real-145), u:[Real](#real-145)) -> [Uniform](#uniform-949)

<a name="uniform-953"></a>

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

<a name="adjacent-difference-1346"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-145)\[\_\]) -> [Integer](#integer-22)

<a name="ancestor-1363"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-145)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="ancestors-1357"></a>

Sample an ancestry vector for a vector of log-weights.

#### beta(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="beta-1177"></a>

The beta function.

#### beta(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="beta-1180"></a>

The beta function.

#### choose(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="choose-1189"></a>

The binomial coefficient.

#### choose(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="choose-1192"></a>

The binomial coefficient.

#### columns(X:[Real](#real-145)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="columns-1091"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-22)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="columns-1093"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-22)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="cumulative-offspring-to-ancestors-1378"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="cumulative-weights-1390"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)

<a name="determinant-1070"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="exclusive-prefix-sum-1342"></a>

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

<a name="gamma-1168"></a>

The gamma function.

#### gamma(x:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="gamma-1170"></a>

The gamma function.

#### identity(rows:[Integer](#integer-22), columns:[Integer](#integer-22)) -> [Real](#real-145)\[\_,\_\]

<a name="identity-1108"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="inclusive-prefix-sum-1338"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="inverse-1074"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-156)) -> [Boolean](#boolean-1)

<a name="isnan-213"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-214)) -> [Boolean](#boolean-1)

<a name="isnan-271"></a>

Does this have the value NaN?

#### lbeta(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="lbeta-1183"></a>

Logarithm of the beta function.

#### lbeta(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="lbeta-1186"></a>

Logarithm of the beta function.

#### lchoose(x:[Real64](#real64-214), y:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="lchoose-1195"></a>

Logarithm of the binomial coefficient.

#### lchoose(x:[Real32](#real32-156), y:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="lchoose-1198"></a>

Logarithm of the binomial coefficient.

#### length(x:[String](#string-272)) -> [Integer](#integer-22)

<a name="length-337"></a>

Length of a string.

#### length(x:[Real](#real-145)\[\_\]) -> [Integer64](#integer64-89)

<a name="length-1308"></a>

Length of a vector.

#### length(x:[Integer](#integer-22)\[\_\]) -> [Integer64](#integer64-89)

<a name="length-1310"></a>

Length of a vector.

#### lgamma(x:[Real64](#real64-214)) -> [Real64](#real64-214)

<a name="lgamma-1172"></a>

Logarithm of the gamma function.

#### lgamma(x:[Real32](#real32-156)) -> [Real32](#real32-156)

<a name="lgamma-1174"></a>

Logarithm of the gamma function.

#### llt(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="llt-1077"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="log-sum-exp-1334"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-145), rows:[Integer](#integer-22), columns:[Integer](#integer-22)) -> [Real](#real-145)\[\_,\_\]

<a name="matrix-1102"></a>

Create matrix filled with a given scalar.

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

<a name="max-1325"></a>

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

<a name="min-1329"></a>

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

<a name="norm-1066"></a>

Norm of a vector.

#### observe_bernoulli(x:[Boolean](#boolean-1), ρ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-bernoulli-1111"></a>

Observe a Bernoulli variate.

- x: The variate.
- ρ: Probability of a true result.

Returns the log probability mass.

#### observe_beta(x:[Real](#real-145), α:[Real](#real-145), β:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-beta-1139"></a>

Observe a Beta variate.

- x: The variate.
- α: Shape.
- β: Shape.

Returns the log probability density.

#### observe_binomial(x:[Integer](#integer-22), n:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-binomial-1115"></a>

Observe a Binomial variate.

- x: The variate.
- n: Number of trials.
- ρ: Probability of a true result.

Returns the log probability mass.

#### observe_gamma(x:[Real](#real-145), k:[Real](#real-145), θ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-gamma-1135"></a>

Observe a Gamma variate.

- x: The variate.
- k: Shape.
- θ: Scale.

Returns the log probability density.

#### observe_gaussian(x:[Real](#real-145), μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-gaussian-1127"></a>

Observe a Gaussian variate.

- x: The variate.
- μ: Mean.
- σ2: Variance.

Returns the log probability density.

#### observe_log_gaussian(x:[Real](#real-145), μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-log-gaussian-1131"></a>

Observe a log-Gaussian variate.

- x: The variate.
- μ: Mean.
- σ2: Variance.

Returns the log probability density.

#### observe_negative_binomial(x:[Integer](#integer-22), k:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-negative-binomial-1119"></a>

Observe a Negative Binomial variate.

- x: The variate (number of failures).
- k: Number of successes before the experiment is stopped.
- ρ: Probability of success.

Returns the log probability mass.

#### observe_uniform(x:[Real](#real-145), l:[Real](#real-145), u:[Real](#real-145)) -> [Real](#real-145)

<a name="observe-uniform-1123"></a>

Observe a Uniform variate.

- x: The variate.
- l: Lower bound of interval.
- u: Upper bound of interval.

Returns the log probability density.

#### permute_ancestors(a:[Integer](#integer-22)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="permute-ancestors-1384"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### read(file:[String](#string-272), N:[Integer](#integer-22)) -> [Real](#real-145)\[\_\]

<a name="read-1352"></a>

Read numbers from a file.

#### rows(X:[Real](#real-145)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="rows-1087"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-22)\[\_,\_\]) -> [Integer64](#integer64-89)

<a name="rows-1089"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)

<a name="scalar-1095"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="scalar-1312"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-22))

<a name="seed-1141"></a>

Seed the pseudorandom number generator.

- seed: Seed value.

#### simulate_bernoulli(ρ:[Real](#real-145)) -> [Boolean](#boolean-1)

<a name="simulate-bernoulli-1143"></a>

Simulate a Bernoulli variate.

- ρ: Probability of a true result.

#### simulate_beta(α:[Real](#real-145), β:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-beta-1166"></a>

Simulate a Beta variate.

- α: Shape.
- β: Shape.

#### simulate_binomial(n:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Integer](#integer-22)

<a name="simulate-binomial-1146"></a>

Simulate a Binomial variate.

- n: Number of trials.
- ρ: Probability of a true result.

#### simulate_gamma(k:[Real](#real-145), θ:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-gamma-1161"></a>

Simulate a Gamma variate.

- k: Shape.
- θ: Scale.

#### simulate_gaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-gaussian-1155"></a>

Simulate a Gaussian variate.

- μ: Mean.
- σ2: Variance.

#### simulate_log_gaussian(μ:[Real](#real-145), σ2:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-log-gaussian-1158"></a>

Simulate a log-Gaussian variate.

- μ: Mean (in log space).
- σ2: Variance (in log space).

#### simulate_negative_binomial(k:[Integer](#integer-22), ρ:[Real](#real-145)) -> [Integer](#integer-22)

<a name="simulate-negative-binomial-1149"></a>

Simulate a Negative Binomial variate.

- k: Number of successes before the experiment is stopped.
- ρ: Probability of success.

Returns the number of failures.

#### simulate_uniform(l:[Real](#real-145), u:[Real](#real-145)) -> [Real](#real-145)

<a name="simulate-uniform-1152"></a>

Simulate a Uniform variate.

- l: Lower bound of interval.
- u: Upper bound of interval.

#### solve(X:[Real](#real-145)\[\_,\_\], y:[Real](#real-145)\[\_\]) -> [Real](#real-145)\[\_\]

<a name="solve-1081"></a>

Solve a system of equations.

#### solve(X:[Real](#real-145)\[\_,\_\], Y:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="solve-1085"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="squarednorm-1068"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-145)\[\_\]) -> [Real](#real-145)

<a name="sum-1321"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-145)\[\_\]) -> [Integer](#integer-22)\[\_\]

<a name="systematic-cumulative-offspring-1370"></a>

Systematic resampling.

#### transpose(X:[Real](#real-145)\[\_,\_\]) -> [Real](#real-145)\[\_,\_\]

<a name="transpose-1072"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-145), length:[Integer](#integer-22)) -> [Real](#real-145)\[\_\]

<a name="vector-1317"></a>

Create vector filled with a given scalar.


# Program Details

#### build(include_dir:[String](#string-272), lib_dir:[String](#string-272), share_dir:[String](#string-272), prefix:[String](#string-272), warnings:[Boolean](#boolean-1) <- true, debug:[Boolean](#boolean-1) <- true, verbose:[Boolean](#boolean-1) <- true)

<a name="build-961"></a>

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

<a name="check-962"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-963"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-964"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-965"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-272) <- "untitled")

<a name="init-967"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-968"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-969"></a>

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

#### type Integer32;


<a name="integer32-33"></a>

A 32-bit integer.

#### type Integer64;


<a name="integer64-89"></a>

A 64-bit integer.

#### type Real32;


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

#### type LogNormal = [LogGaussian](#loggaussian-890);


<a name="lognormal-891"></a>

Synonym for LogGaussian.

#### type MultivariateNormal = [MultivariateGaussian](#multivariategaussian-913);


<a name="multivariatenormal-914"></a>

Synonym for MultivariateGaussian.

#### type Normal = [Gaussian](#gaussian-868);


<a name="normal-869"></a>

Synonym for Gaussian.

#### type Real = [Real64](#real64-214);


<a name="real-145"></a>

A floating point value of default type.


# Class Type Details


## Bernoulli

<a name="bernoulli-814"></a>

  * Inherits from *[DelayBoolean](#delayboolean-560)*

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-145)* | Probability of a true result. |


## Beta

<a name="beta-800"></a>

  * Inherits from *[DelayReal](#delayreal-590)*

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-145)* | First shape parameter. |
| *β:[Real](#real-145)* | Second shape parameter. |


## Binomial

<a name="binomial-830"></a>

  * Inherits from *[DelayInteger](#delayinteger-575)*

Binomial distribution.

| Member Variable | Description |
| --- | --- |
| *n:[Integer](#integer-22)* | Number of trials. |
| *ρ:[Real](#real-145)* | Probability of a true result. |


## Delay

<a name="delay-545"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-22)* | State of the variate. |
| *missing:[Boolean](#boolean-1)* | Is the value missing? |
| *parent:[Delay](#delay-545)?* | Parent. |
| *child:[Delay](#delay-545)?* | Child, if one exists and it is on the stem. |
| *id:[Integer](#integer-22)* | Unique id for delayed sampling diagnostics. |
| *nforward:[Integer](#integer-22)* | Number of observations absorbed in forward pass, for delayed sampling diagnostics. |
| *nbackward:[Integer](#integer-22)* | Number of observations absorbed in backward pass, for delayed sampling diagnostics. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-512) | Is this a root node? |
| [isTerminal](#isterminal-513) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-514) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-515) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-516) | Is this node in the marginalized state? |
| [isRealized](#isrealized-517) | Is this node in the realized state? |
| [isMissing](#ismissing-518) | Is the value of this node missing? |
| [isNotMissing](#isnotmissing-519) | Is the value of this node not missing? |
| [initialize](#initialize-520) | Initialize as a root node. |
| [initialize](#initialize-522) | Initialize as a non-root node. |
| [absorb](#absorb-524) | Increment number of observations absorbed. |
| [marginalize](#marginalize-525) | Marginalize the variate. |
| [simulate](#simulate-526) | Simulate the variate. |
| [observe](#observe-527) | Observe the variate. |
| [realize](#realize-528) | Realize the variate. |
| [graft](#graft-529) | Graft the stem to this node. |
| [graft](#graft-531) | Graft the stem to this node. |
| [prune](#prune-532) | Prune the stem from below this node. |
| [setParent](#setparent-534) | Set the parent. |
| [removeParent](#removeparent-535) | Remove the parent. |
| [setChild](#setchild-537) | Set the child. |
| [removeChild](#removechild-538) | Remove the child. |
| [register](#register-543) | Register with the diagnostic handler. |
| [trigger](#trigger-544) | Trigger an event with the diagnostic handler. |


### Member Function Details

#### absorb(nbackward:[Integer](#integer-22))

<a name="absorb-524"></a>

Increment number of observations absorbed.

  - `nbackward` : Number of new observations absorbed.

#### graft()

<a name="graft-529"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-545))

<a name="graft-531"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-520"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-545))

<a name="initialize-522"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-1)

<a name="isinitialized-515"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-1)

<a name="ismarginalized-516"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-1)

<a name="ismissing-518"></a>

Is the value of this node missing?

#### isNotMissing() -> [Boolean](#boolean-1)

<a name="isnotmissing-519"></a>

Is the value of this node not missing?

#### isRealized() -> [Boolean](#boolean-1)

<a name="isrealized-517"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-1)

<a name="isroot-512"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-1)

<a name="isterminal-513"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-1)

<a name="isuninitialized-514"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-525"></a>

Marginalize the variate.

#### observe()

<a name="observe-527"></a>

Observe the variate.

#### prune()

<a name="prune-532"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-528"></a>

Realize the variate.

#### register()

<a name="register-543"></a>

Register with the diagnostic handler.

#### removeChild()

<a name="removechild-538"></a>

Remove the child.

#### removeParent()

<a name="removeparent-535"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-545))

<a name="setchild-537"></a>

Set the child.

#### setParent(u:[Delay](#delay-545))

<a name="setparent-534"></a>

Set the parent.

#### simulate()

<a name="simulate-526"></a>

Simulate the variate.

#### trigger()

<a name="trigger-544"></a>

Trigger an event with the diagnostic handler.


## DelayBoolean

<a name="delayboolean-560"></a>

  * Inherits from *[Delay](#delay-545)*

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
| *nodes:[Delay](#delay-545)?\[\_\]* | Registered nodes. |
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

#### register(o:[Delay](#delay-545)) -> [Integer](#integer-22)

<a name="register-347"></a>

Register a new node. This is a callback function typically called
within the Delay class itself.

Returns an id assigned to the node.

#### trigger()

<a name="trigger-357"></a>

Trigger an event.


## DelayInteger

<a name="delayinteger-575"></a>

  * Inherits from *[Delay](#delay-545)*

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

<a name="delayreal-590"></a>

  * Inherits from *[Delay](#delay-545)*

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

<a name="delayrealvector-604"></a>

  * Inherits from *[Delay](#delay-545)*

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


## FileOutputStream

<a name="fileoutputstream-972"></a>

  * Inherits from *[OutputStream](#outputstream-1008)*

File output stream.

| Member Function | Brief description |
| --- | --- |
| [close](#close-971) | Close the file. |


### Member Function Details

#### close()

<a name="close-971"></a>

Close the file.


## Gamma

<a name="gamma-847"></a>

  * Inherits from *[DelayReal](#delayreal-590)*

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-145)* | Shape. |
| *θ:[Real](#real-145)* | Scale. |


## Gaussian

<a name="gaussian-868"></a>

  * Inherits from *[DelayReal](#delayreal-590)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-145)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |


## GaussianExpression

<a name="gaussianexpression-612"></a>

Expression that is an affine transformation of a Gaussian variable.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-145)* | Scale. |
| *x:[Gaussian](#gaussian-868)* | Base distribution. |
| *c:[Real](#real-145)* | Offset. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-611) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-145), x:[Gaussian](#gaussian-868), c:[Real](#real-145))

<a name="initialize-611"></a>

Initialize.


## GaussianLogExpression

<a name="gaussianlogexpression-680"></a>

Expression that is an affine transformation of the logarithm of a
log-Gaussian variable.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-145)* | Scale. |
| *x:[LogGaussian](#loggaussian-890)* | Base distribution. |
| *c:[Real](#real-145)* | Offset. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-679) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-145), x:[LogGaussian](#loggaussian-890), c:[Real](#real-145))

<a name="initialize-679"></a>

Initialize.


## GaussianWithGaussianExpressionMean

<a name="gaussianwithgaussianexpressionmean-379"></a>

  * Inherits from *[Gaussian](#gaussian-868)*

Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianExpression](#gaussianexpression-612)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## GaussianWithGaussianLogExpressionMean

<a name="gaussianwithgaussianlogexpressionmean-395"></a>

  * Inherits from *[Gaussian](#gaussian-868)*

Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianLogExpression](#gaussianlogexpression-680)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## GaussianWithGaussianMean

<a name="gaussianwithgaussianmean-411"></a>

  * Inherits from *[Gaussian](#gaussian-868)*

Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[Gaussian](#gaussian-868)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## LogGaussian

<a name="loggaussian-890"></a>

  * Inherits from *[DelayReal](#delayreal-590)*

Log-Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-145)* | Mean after log transformation. |
| *σ2:[Real](#real-145)* | Variance after log transformation. |


## LogGaussianExpression

<a name="loggaussianexpression-716"></a>

Expression that is a scaling of a log-Gaussian variable.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-145)* | Scale. |
| *x:[LogGaussian](#loggaussian-890)* | Base distribution. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-715) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-145), x:[LogGaussian](#loggaussian-890))

<a name="initialize-715"></a>

Initialize.


## LogGaussianWithGaussianExpressionMean

<a name="loggaussianwithgaussianexpressionmean-427"></a>

  * Inherits from *[LogGaussian](#loggaussian-890)*

Log-Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianExpression](#gaussianexpression-612)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## LogGaussianWithGaussianLogExpressionMean

<a name="loggaussianwithgaussianlogexpressionmean-443"></a>

  * Inherits from *[LogGaussian](#loggaussian-890)*

Log-Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[GaussianLogExpression](#gaussianlogexpression-680)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## LogGaussianWithGaussianMean

<a name="loggaussianwithgaussianmean-459"></a>

  * Inherits from *[LogGaussian](#loggaussian-890)*

Log-Gaussian with conjugate prior on mean.

| Member Variable | Description |
| --- | --- |
| *μ:[Gaussian](#gaussian-868)* | Mean. |
| *σ2:[Real](#real-145)* | Variance. |
| *μ_0:[Real](#real-145)* | Prior (marginalized) mean. |
| *σ2_0:[Real](#real-145)* | Prior (marginalized) variance. |


## MultivariateGaussian

<a name="multivariategaussian-913"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-604)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-145)\[\_\]* | Mean. |
| *Σ:[Real](#real-145)\[\_,\_\]* | Covariance. |


## MultivariateGaussianExpression

<a name="multivariategaussianexpression-749"></a>

Expression that is an affine transformation of a multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Conversion | Description |
| --- | --- |
| *[Real](#real-145)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-145)\[\_,\_\]* | Scale. |
| *x:[MultivariateGaussian](#multivariategaussian-913)* | Base distribution. |
| *c:[Real](#real-145)\[\_\]* | Offset. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-748) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-145)\[\_,\_\], x:[MultivariateGaussian](#multivariategaussian-913), c:[Real](#real-145)\[\_\])

<a name="initialize-748"></a>

Initialize.


## MultivariateGaussianWithMultivariateGaussianExpressionMean

<a name="multivariategaussianwithmultivariategaussianexpressionmean-479"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-913)*

Multivariate Gaussian with conjugate prior on mean.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[MultivariateGaussianExpression](#multivariategaussianexpression-749)?* | Mean. |
| *Σ:[Real](#real-145)\[\_,\_\]* | Covariance. |
| *μ_0:[Real](#real-145)\[\_\]* | Marginalized prior mean. |
| *Σ_0:[Real](#real-145)\[\_,\_\]* | Marginalized prior covariance. |


## MultivariateGaussianWithMultivariateGaussianMean

<a name="multivariategaussianwithmultivariategaussianmean-496"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-913)*

Multivariate Gaussian with conjugate prior on mean.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[MultivariateGaussian](#multivariategaussian-913)* | Mean. |
| *Σ:[Real](#real-145)\[\_,\_\]* | Covariance. |
| *μ_0:[Real](#real-145)\[\_\]* | Marginalized prior mean. |
| *Σ_0:[Real](#real-145)\[\_,\_\]* | Marginalized prior covariance. |


## NegativeBinomial

<a name="negativebinomial-932"></a>

  * Inherits from *[DelayInteger](#delayinteger-575)*

Negative binomial distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Integer](#integer-22)* | Number of successes before the experiment is stopped. |
| *ρ:[Real](#real-145)* | Probability of success. |


## OutputStream

<a name="outputstream-1008"></a>

Output stream.

| Member Function | Brief description |
| --- | --- |
| [printf](#printf-976) | Print with format. |
| [printf](#printf-979) | Print with format. |
| [printf](#printf-982) | Print with format. |
| [printf](#printf-985) | Print with format. |
| [print](#print-987) | Print scalar. |
| [print](#print-989) | Print scalar. |
| [print](#print-991) | Print scalar. |
| [print](#print-993) | Print scalar. |
| [print](#print-996) | Print vector. |
| [print](#print-999) | Print vector. |
| [print](#print-1003) | Print matrix. |
| [print](#print-1007) | Print matrix. |


### Member Function Details

#### print(value:[Boolean](#boolean-1))

<a name="print-987"></a>

Print scalar.

#### print(value:[Integer](#integer-22))

<a name="print-989"></a>

Print scalar.

#### print(value:[Real](#real-145))

<a name="print-991"></a>

Print scalar.

#### print(value:[String](#string-272))

<a name="print-993"></a>

Print scalar.

#### print(x:[Integer](#integer-22)\[\_\])

<a name="print-996"></a>

Print vector.

#### print(x:[Real](#real-145)\[\_\])

<a name="print-999"></a>

Print vector.

#### print(X:[Integer](#integer-22)\[\_,\_\])

<a name="print-1003"></a>

Print matrix.

#### print(X:[Real](#real-145)\[\_,\_\])

<a name="print-1007"></a>

Print matrix.

#### printf(fmt:[String](#string-272), value:[Boolean](#boolean-1))

<a name="printf-976"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-272), value:[Integer](#integer-22))

<a name="printf-979"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-272), value:[Real](#real-145))

<a name="printf-982"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-272), value:[String](#string-272))

<a name="printf-985"></a>

Print with format. See system `printf`.


## StdErrStream

<a name="stderrstream-1009"></a>

  * Inherits from *[OutputStream](#outputstream-1008)*

Output stream for stderr.


## StdOutStream

<a name="stdoutstream-1012"></a>

  * Inherits from *[OutputStream](#outputstream-1008)*

Output stream for stdout.


## Uniform

<a name="uniform-949"></a>

  * Inherits from *[DelayReal](#delayreal-590)*

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-145)* | Lower bound. |
| *u:[Real](#real-145)* | Upper bound. |

