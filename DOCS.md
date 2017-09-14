
# Summary

| Variable | Description |
| --- | --- |
| *delayDiagnostics:[DelayDiagnostics](#delaydiagnostics-431)?* | Global diagnostics handler for delayed sampling. |
| *inf:[Real64](#real64-681)* | $\infty$ |
| *stderr:[StdErrStream](#stderrstream-124)* | Standard error. |
| *stdout:[StdOutStream](#stdoutstream-113)* | Standard output. |
| *π:[Real64](#real64-681)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-283) | Create. |
| [Beta](#beta-517) | Create a Beta distribution. |
| [Boolean](#boolean-986) | Convert other basic types to Boolean. |
| [Gamma](#gamma-497) | Create Gamma distribution. |
| [Gaussian](#gaussian-270) | Create a Gaussian distribution. |
| [Gaussian](#gaussian-454) | Create. |
| [Integer](#integer-670) | Convert other basic types to Integer. |
| [Integer32](#integer32-805) | Convert other basic types to Integer32. |
| [Integer64](#integer64-306) | Convert other basic types to Integer64. |
| [Real](#real-867) | Convert other basic types to Real. |
| [Real32](#real32-41) | Convert other basic types to Real32. |
| [Real64](#real64-683) | Convert other basic types to Real64. |
| [String](#string-999) | Convert other basic types to String. |
| [Uniform](#uniform-251) | Create a Uniform distribution. |
| [abs](#abs-849) | Absolute value. |
| [abs](#abs-350) | Absolute value. |
| [abs](#abs-85) | Absolute value. |
| [abs](#abs-727) | Absolute value. |
| [adjacent_difference](#adjacent-difference-234) | Inclusive prefix sum. |
| [ancestor](#ancestor-11) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-5) | Sample an ancestry vector for a vector of log-weights. |
| [beta](#beta-294) | The beta function. |
| [beta](#beta-297) | The beta function. |
| [columns](#columns-184) | Number of columns of a matrix. |
| [columns](#columns-186) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-26) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-38) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-652) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-230) | Inclusive prefix sum. |
| [fclose](#fclose-477) | Close a file. |
| [fopen](#fopen-472) | Open a file for reading. |
| [fopen](#fopen-475) | Open a file. |
| [gamma](#gamma-285) | The gamma function. |
| [gamma](#gamma-287) | The gamma function. |
| [identity](#identity-201) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-226) | Inclusive prefix sum. |
| [inverse](#inverse-656) | Inverse of a matrix. |
| [isnan](#isnan-96) | Does this have the value NaN? |
| [isnan](#isnan-738) | Does this have the value NaN? |
| [lbeta](#lbeta-300) | Logarithm of the beta function. |
| [lbeta](#lbeta-303) | Logarithm of the beta function. |
| [length](#length-1062) | Length of a string. |
| [length](#length-519) | Length of a vector. |
| [length](#length-521) | Length of a vector. |
| [lgamma](#lgamma-289) | Logarithm of the gamma function. |
| [lgamma](#lgamma-291) | Logarithm of the gamma function. |
| [llt](#llt-659) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-222) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-195) | Create matrix filled with a given scalar. |
| [max](#max-855) | Maximum of two values. |
| [max](#max-356) | Maximum of two values. |
| [max](#max-91) | Maximum of two values. |
| [max](#max-733) | Maximum of two values. |
| [max](#max-213) | Maximum of a vector. |
| [min](#min-858) | Minimum of two values. |
| [min](#min-359) | Minimum of two values. |
| [min](#min-94) | Minimum of two values. |
| [min](#min-736) | Minimum of two values. |
| [min](#min-217) | Minimum of a vector. |
| [mod](#mod-852) | Modulus. |
| [mod](#mod-353) | Modulus. |
| [mod](#mod-88) | Modulus. |
| [mod](#mod-730) | Modulus. |
| [norm](#norm-648) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-32) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [read](#read-864) | Read numbers from a file. |
| [rows](#rows-180) | Number of rows of a matrix. |
| [rows](#rows-182) | Number of rows of a matrix. |
| [scalar](#scalar-188) | Convert single-element matrix to scalar. |
| [scalar](#scalar-523) | Convert single-element vector to scalar. |
| [seed](#seed-98) | Seed the pseudorandom number generator. |
| [simulate_bernoulli](#simulate-bernoulli-100) | Simulate a Bernoulli variate. |
| [simulate_binomial](#simulate-binomial-103) | Simulate a Binomial variate. |
| [simulate_gamma](#simulate-gamma-112) | Simulate a Gamma variate. |
| [simulate_gaussian](#simulate-gaussian-109) | Simulate a Gaussian variate. |
| [simulate_uniform](#simulate-uniform-106) | Simulate a Uniform variate. |
| [solve](#solve-663) | Solve a system of equations. |
| [solve](#solve-667) | Solve a system of equations. |
| [squaredNorm](#squarednorm-650) | Squared norm of a vector. |
| [sum](#sum-209) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-18) | Systematic resampling. |
| [transpose](#transpose-654) | Transpose of a matrix. |
| [vector](#vector-528) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-123) | Build the project. |
| [check](#check-205) | Check the file structure of the project for possible issues. |
| [clean](#clean-544) | Clean the project directory of all build files. |
| [dist](#dist-204) | Build a distributable archive for the project. |
| [docs](#docs-203) | Build the reference documentation for the project. |
| [init](#init-680) | Initialise the working directory for a new project. |
| [install](#install-202) | Install the project. |
| [uninstall](#uninstall-178) | Uninstall the project. |

| Basic Type | Brief description |
| --- | --- |
| [Boolean](#boolean-984) | A Boolean value. |
| [File](#file-470) | A file handle. |
| [Integer32](#integer32-803) | A 32-bit integer. |
| [Integer64](#integer64-304) | A 64-bit integer. |
| [Real32](#real32-39) | A 32-bit (single precision) floating point value. |
| [Real64](#real64-681) | A 64-bit (double precision) floating point value. |
| [String](#string-997) | A string value. |

| Alias Type | Brief description |
| --- | --- |
| [Integer](#integer-668) | An integer value of default type. |
| [Real](#real-865) | A floating point value of default type. |

| Class Type | Brief description |
| --- | --- |
| [AffineGaussian](#affinegaussian-754) | Gaussian that has a mean which is an affine transformation of another Gaussian. |
| [AffineGaussianExpression](#affinegaussianexpression-548) | Expression used to accumulate affine transformations of Gaussians. |
| [AffineMultivariateGaussian](#affinemultivariategaussian-794) | Multivariate Gaussian that has a mean which is an affine transformation of another multivariate Gaussian. |
| [AffineMultivariateGaussianExpression](#affinemultivariategaussianexpression-363) | Expression used to accumulate affine transformations of multivariate Gaussians. |
| [Bernoulli](#bernoulli-280) | Bernoulli distribution. |
| [Beta](#beta-513) | Beta distribution. |
| [Delay](#delay-1107) | Node interface for delayed sampling. |
| [DelayBoolean](#delayboolean-177) | Abstract delay variate with Boolean value. |
| [DelayDiagnostics](#delaydiagnostics-431) | Outputs graphical representations of the delayed sampling state for diagnostic purposes. |
| [DelayInteger](#delayinteger-543) | Abstract delay variate with Integer value. |
| [DelayReal](#delayreal-469) | Abstract delay variate with Real value. |
| [DelayRealVector](#delayrealvector-776) | Abstract delay variate with real vector value. |
| [FileOutputStream](#fileoutputstream-480) | File output stream. |
| [Gamma](#gamma-493) | Gamma distribution. |
| [Gaussian](#gaussian-266) | Gaussian distribution. |
| [MultivariateGaussian](#multivariategaussian-449) | Multivariate Gaussian distribution. |
| [OutputStream](#outputstream-162) | Output stream. |
| [StdErrStream](#stderrstream-124) | Output stream for stderr. |
| [StdOutStream](#stdoutstream-113) | Output stream for stdout. |
| [Uniform](#uniform-247) | Uniform distribution. |


# Function Details

#### Bernoulli(ρ:[Real](#real-865)) -> [Bernoulli](#bernoulli-280)

<a name="bernoulli-283"></a>

Create.

#### Beta(α:[Real](#real-865), β:[Real](#real-865)) -> [Beta](#beta-513)

<a name="beta-517"></a>

Create a Beta distribution.

#### Boolean(x:[Boolean](#boolean-984)) -> [Boolean](#boolean-984)

<a name="boolean-986"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-865), θ:[Real](#real-865)) -> [Gamma](#gamma-493)

<a name="gamma-497"></a>

Create Gamma distribution.

#### Gaussian(μ:[Real](#real-865), σ2:[Real](#real-865)) -> [Gaussian](#gaussian-266)

<a name="gaussian-270"></a>

Create a Gaussian distribution.

#### Gaussian(μ:[Real](#real-865)\[\_\], Σ:[Real](#real-865)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-449)

<a name="gaussian-454"></a>

Create.

#### Integer(x:[Integer64](#integer64-304)) -> [Integer](#integer-668)

<a name="integer-670"></a>

Convert other basic types to Integer. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer32(x:[Integer32](#integer32-803)) -> [Integer32](#integer32-803)

<a name="integer32-805"></a>

Convert other basic types to Integer32. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer64(x:[Integer64](#integer64-304)) -> [Integer64](#integer64-304)

<a name="integer64-306"></a>

Convert other basic types to Integer64. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real(x:[Real64](#real64-681)) -> [Real](#real-865)

<a name="real-867"></a>

Convert other basic types to Real. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real32(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="real32-41"></a>

Convert other basic types to Real32. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### Real64(x:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="real64-683"></a>

Convert other basic types to Real64. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### String(x:[String](#string-997)) -> [String](#string-997)

<a name="string-999"></a>

Convert other basic types to String. This is overloaded for Bolean, Real64,
String, Integer64, Integer32 and String.

#### Uniform(l:[Real](#real-865), u:[Real](#real-865)) -> [Uniform](#uniform-247)

<a name="uniform-251"></a>

Create a Uniform distribution.

#### abs(x:[Integer32](#integer32-803)) -> [Integer32](#integer32-803)

<a name="abs-849"></a>

Absolute value.

#### abs(x:[Integer64](#integer64-304)) -> [Integer64](#integer64-304)

<a name="abs-350"></a>

Absolute value.

#### abs(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="abs-85"></a>

Absolute value.

#### abs(x:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="abs-727"></a>

Absolute value.

#### adjacent_difference(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)\[\_\]

<a name="adjacent-difference-234"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-865)\[\_\]) -> [Integer](#integer-668)

<a name="ancestor-11"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-865)\[\_\]) -> [Integer](#integer-668)\[\_\]

<a name="ancestors-5"></a>

Sample an ancestry vector for a vector of log-weights.

#### beta(x:[Real64](#real64-681), y:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="beta-294"></a>

The beta function.

#### beta(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="beta-297"></a>

The beta function.

#### columns(X:[Real](#real-865)\[\_,\_\]) -> [Integer64](#integer64-304)

<a name="columns-184"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-668)\[\_,\_\]) -> [Integer64](#integer64-304)

<a name="columns-186"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-668)\[\_\]) -> [Integer](#integer-668)\[\_\]

<a name="cumulative-offspring-to-ancestors-26"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-865)\[\_\]) -> [Real](#real-865)\[\_\]

<a name="cumulative-weights-38"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-865)\[\_,\_\]) -> [Real](#real-865)

<a name="determinant-652"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)\[\_\]

<a name="exclusive-prefix-sum-230"></a>

Inclusive prefix sum.

#### fclose(file:[File](#file-470))

<a name="fclose-477"></a>

Close a file.

#### fopen(file:[String](#string-997)) -> [File](#file-470)

<a name="fopen-472"></a>

Open a file for reading.

  - file : The file name.

#### fopen(file:[String](#string-997), mode:[String](#string-997)) -> [File](#file-470)

<a name="fopen-475"></a>

Open a file.

  - file : The file name.
  - mode : The mode, either `r` (read), `w` (write), `a` (append) or any
    other modes as in system `fopen`.

#### gamma(x:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="gamma-285"></a>

The gamma function.

#### gamma(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="gamma-287"></a>

The gamma function.

#### identity(rows:[Integer](#integer-668), columns:[Integer](#integer-668)) -> [Real](#real-865)\[\_,\_\]

<a name="identity-201"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)\[\_\]

<a name="inclusive-prefix-sum-226"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-865)\[\_,\_\]) -> [Real](#real-865)\[\_,\_\]

<a name="inverse-656"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-39)) -> [Boolean](#boolean-984)

<a name="isnan-96"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-681)) -> [Boolean](#boolean-984)

<a name="isnan-738"></a>

Does this have the value NaN?

#### lbeta(x:[Real64](#real64-681), y:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="lbeta-300"></a>

Logarithm of the beta function.

#### lbeta(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="lbeta-303"></a>

Logarithm of the beta function.

#### length(x:[String](#string-997)) -> [Integer](#integer-668)

<a name="length-1062"></a>

Length of a string.

#### length(x:[Real](#real-865)\[\_\]) -> [Integer64](#integer64-304)

<a name="length-519"></a>

Length of a vector.

#### length(x:[Integer](#integer-668)\[\_\]) -> [Integer64](#integer64-304)

<a name="length-521"></a>

Length of a vector.

#### lgamma(x:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="lgamma-289"></a>

Logarithm of the gamma function.

#### lgamma(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="lgamma-291"></a>

Logarithm of the gamma function.

#### llt(X:[Real](#real-865)\[\_,\_\]) -> [Real](#real-865)\[\_,\_\]

<a name="llt-659"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)

<a name="log-sum-exp-222"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-865), rows:[Integer](#integer-668), columns:[Integer](#integer-668)) -> [Real](#real-865)\[\_,\_\]

<a name="matrix-195"></a>

Create matrix filled with a given scalar.

#### max(x:[Integer32](#integer32-803), y:[Integer32](#integer32-803)) -> [Integer32](#integer32-803)

<a name="max-855"></a>

Maximum of two values.

#### max(x:[Integer64](#integer64-304), y:[Integer64](#integer64-304)) -> [Integer64](#integer64-304)

<a name="max-356"></a>

Maximum of two values.

#### max(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="max-91"></a>

Maximum of two values.

#### max(x:[Real64](#real64-681), y:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="max-733"></a>

Maximum of two values.

#### max(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)

<a name="max-213"></a>

Maximum of a vector.

#### min(x:[Integer32](#integer32-803), y:[Integer32](#integer32-803)) -> [Integer32](#integer32-803)

<a name="min-858"></a>

Minimum of two values.

#### min(x:[Integer64](#integer64-304), y:[Integer64](#integer64-304)) -> [Integer64](#integer64-304)

<a name="min-359"></a>

Minimum of two values.

#### min(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="min-94"></a>

Minimum of two values.

#### min(x:[Real64](#real64-681), y:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="min-736"></a>

Minimum of two values.

#### min(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)

<a name="min-217"></a>

Minimum of a vector.

#### mod(x:[Integer32](#integer32-803), y:[Integer32](#integer32-803)) -> [Integer32](#integer32-803)

<a name="mod-852"></a>

Modulus.

#### mod(x:[Integer64](#integer64-304), y:[Integer64](#integer64-304)) -> [Integer64](#integer64-304)

<a name="mod-353"></a>

Modulus.

#### mod(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="mod-88"></a>

Modulus.

#### mod(x:[Real64](#real64-681), y:[Real64](#real64-681)) -> [Real64](#real64-681)

<a name="mod-730"></a>

Modulus.

#### norm(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)

<a name="norm-648"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-668)\[\_\]) -> [Integer](#integer-668)\[\_\]

<a name="permute-ancestors-32"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### read(file:[String](#string-997), N:[Integer](#integer-668)) -> [Real](#real-865)\[\_\]

<a name="read-864"></a>

Read numbers from a file.

#### rows(X:[Real](#real-865)\[\_,\_\]) -> [Integer64](#integer64-304)

<a name="rows-180"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-668)\[\_,\_\]) -> [Integer64](#integer64-304)

<a name="rows-182"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-865)\[\_,\_\]) -> [Real](#real-865)

<a name="scalar-188"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)

<a name="scalar-523"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-668))

<a name="seed-98"></a>

Seed the pseudorandom number generator.

- seed: Seed value.

#### simulate_bernoulli(ρ:[Real](#real-865)) -> [Boolean](#boolean-984)

<a name="simulate-bernoulli-100"></a>

Simulate a Bernoulli variate.

- ρ: Probability of a true result.

#### simulate_binomial(n:[Integer](#integer-668), ρ:[Real](#real-865)) -> [Integer](#integer-668)

<a name="simulate-binomial-103"></a>

Simulate a Binomial variate.

- n: Number of trials.
- ρ: Probability of a true result.

#### simulate_gamma(k:[Real](#real-865), θ:[Real](#real-865)) -> [Real](#real-865)

<a name="simulate-gamma-112"></a>

Simulate a Gamma variate.

- k: Shape.
- θ: Scale.

#### simulate_gaussian(μ:[Real](#real-865), σ2:[Real](#real-865)) -> [Real](#real-865)

<a name="simulate-gaussian-109"></a>

Simulate a Gaussian variate.

- μ: Mean.
- σ2: Variance.

#### simulate_uniform(l:[Real](#real-865), u:[Real](#real-865)) -> [Real](#real-865)

<a name="simulate-uniform-106"></a>

Simulate a Uniform variate.

- l: Lower bound of interval.
- u: Upper bound of interval.

#### solve(X:[Real](#real-865)\[\_,\_\], y:[Real](#real-865)\[\_\]) -> [Real](#real-865)\[\_\]

<a name="solve-663"></a>

Solve a system of equations.

#### solve(X:[Real](#real-865)\[\_,\_\], Y:[Real](#real-865)\[\_,\_\]) -> [Real](#real-865)\[\_,\_\]

<a name="solve-667"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)

<a name="squarednorm-650"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-865)\[\_\]) -> [Real](#real-865)

<a name="sum-209"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-865)\[\_\]) -> [Integer](#integer-668)\[\_\]

<a name="systematic-cumulative-offspring-18"></a>

Systematic resampling.

#### transpose(X:[Real](#real-865)\[\_,\_\]) -> [Real](#real-865)\[\_,\_\]

<a name="transpose-654"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-865), length:[Integer](#integer-668)) -> [Real](#real-865)\[\_\]

<a name="vector-528"></a>

Create vector filled with a given scalar.


# Program Details

#### build(include_dir:[String](#string-997), lib_dir:[String](#string-997), share_dir:[String](#string-997), prefix:[String](#string-997), warnings:[Boolean](#boolean-984) <- true, debug:[Boolean](#boolean-984) <- true, verbose:[Boolean](#boolean-984) <- true)

<a name="build-123"></a>

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

<a name="check-205"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-544"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-204"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-203"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-997) <- "untitled")

<a name="init-680"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-202"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-178"></a>

Uninstall the project. This uninstalls all header, library and data files
from the directory specified by `--prefix` (or the system default if this
was not specified).


# Basic Type Details

#### type Boolean

<a name="boolean-984"></a>

A Boolean value.

#### type File

<a name="file-470"></a>

A file handle.

#### type Integer32

<a name="integer32-803"></a>

A 32-bit integer.

#### type Integer64

<a name="integer64-304"></a>

A 64-bit integer.

#### type Real32

<a name="real32-39"></a>

A 32-bit (single precision) floating point value.

#### type Real64

<a name="real64-681"></a>

A 64-bit (double precision) floating point value.

#### type String

<a name="string-997"></a>

A string value.


# Alias Type Details

#### type Integer = [Integer64](#integer64-304)

<a name="integer-668"></a>

An integer value of default type.

#### type Real = [Real64](#real64-681)

<a name="real-865"></a>

A floating point value of default type.


# Class Type Details


## AffineGaussian

<a name="affinegaussian-754"></a>

  * Inherits from *[Gaussian](#gaussian-266)*

Gaussian that has a mean which is an affine transformation of another
Gaussian.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-865)* | Multiplicative scalar of affine transformation. |
| *μ:[Gaussian](#gaussian-266)* | Mean. |
| *c:[Real](#real-865)* | Additive scalar of affine transformation. |
| *q:[Real](#real-865)* | Variance. |
| *y:[Real](#real-865)* | Marginalized prior mean. |
| *s:[Real](#real-865)* | Marginalized prior variance. |


## AffineGaussianExpression

<a name="affinegaussianexpression-548"></a>

Expression used to accumulate affine transformations of Gaussians.

  - `a` Multiplicative scalar of affine transformation.
  - `u` Parent.
  - `c` Additive scalar of affine transformation.

| Conversion | Description |
| --- | --- |
| *[Real](#real-865)* | Value conversion. |


## AffineMultivariateGaussian

<a name="affinemultivariategaussian-794"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-449)*

Multivariate Gaussian that has a mean which is an affine transformation of
another multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-865)\[\_,\_\]* | Matrix of affine transformation. |
| *μ:[MultivariateGaussian](#multivariategaussian-449)* | Mean. |
| *c:[Real](#real-865)\[\_\]* | Vector of affine transformation. |
| *Q:[Real](#real-865)\[\_,\_\]* | Disturbance covariance. |
| *y:[Real](#real-865)\[\_\]* | Marginalized prior mean. |
| *S:[Real](#real-865)\[\_,\_\]* | Marginalized prior covariance. |


## AffineMultivariateGaussianExpression

<a name="affinemultivariategaussianexpression-363"></a>

Expression used to accumulate affine transformations of multivariate
Gaussians.

  - `A` Matrix of affine transformation.
  - `u` Parent.
  - `c` Vector of affine transformation.

| Conversion | Description |
| --- | --- |
| *[Real](#real-865)\[\_\]* | Value conversion. |


## Bernoulli

<a name="bernoulli-280"></a>

  * Inherits from *[DelayBoolean](#delayboolean-177)*

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-865)* | Probability of a true result. |


## Beta

<a name="beta-513"></a>

  * Inherits from *[DelayReal](#delayreal-469)*

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-865)* | First shape parameter. |
| *β:[Real](#real-865)* | Second shape parameter. |


## Delay

<a name="delay-1107"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-668)* | State of the variate. |
| *missing:[Boolean](#boolean-984)* | Is the value missing? |
| *parent:[Delay](#delay-1107)?* | Parent. |
| *child:[Delay](#delay-1107)?* | Child, if one exists and it is on the stem. |
| *id:[Integer](#integer-668)* | Unique id for delayed sampling diagnostics. |
| *nforward:[Integer](#integer-668)* | Number of observations absorbed in forward pass, for delayed sampling diagnostics. |
| *nbackward:[Integer](#integer-668)* | Number of observations absorbed in backward pass, for delayed sampling diagnostics. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-1074) | Is this a root node? |
| [isTerminal](#isterminal-1075) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-1076) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-1077) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-1078) | Is this node in the marginalized state? |
| [isRealized](#isrealized-1079) | Is this node in the realized state? |
| [isMissing](#ismissing-1080) | Is the value of this node missing? |
| [isNotMissing](#isnotmissing-1081) | Is the value of this node not missing? |
| [initialize](#initialize-1082) | Initialize as a root node. |
| [initialize](#initialize-1084) | Initialize as a non-root node. |
| [absorb](#absorb-1086) | Increment number of observations absorbed. |
| [marginalize](#marginalize-1087) | Marginalize the variate. |
| [simulate](#simulate-1088) | Simulate the variate. |
| [observe](#observe-1089) | Observe the variate. |
| [realize](#realize-1090) | Realize the variate. |
| [graft](#graft-1091) | Graft the stem to this node. |
| [graft](#graft-1093) | Graft the stem to this node. |
| [prune](#prune-1094) | Prune the stem from below this node. |
| [setParent](#setparent-1096) | Set the parent. |
| [removeParent](#removeparent-1097) | Remove the parent. |
| [setChild](#setchild-1099) | Set the child. |
| [removeChild](#removechild-1100) | Remove the child. |
| [register](#register-1105) | Register with the diagnostic handler. |
| [trigger](#trigger-1106) | Trigger an event with the diagnostic handler. |


### Member Function Details

#### absorb(nbackward:[Integer](#integer-668))

<a name="absorb-1086"></a>

Increment number of observations absorbed.

  - `nbackward` : Number of new observations absorbed.

#### graft()

<a name="graft-1091"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-1107))

<a name="graft-1093"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-1082"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-1107))

<a name="initialize-1084"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-984)

<a name="isinitialized-1077"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-984)

<a name="ismarginalized-1078"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-984)

<a name="ismissing-1080"></a>

Is the value of this node missing?

#### isNotMissing() -> [Boolean](#boolean-984)

<a name="isnotmissing-1081"></a>

Is the value of this node not missing?

#### isRealized() -> [Boolean](#boolean-984)

<a name="isrealized-1079"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-984)

<a name="isroot-1074"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-984)

<a name="isterminal-1075"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-984)

<a name="isuninitialized-1076"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-1087"></a>

Marginalize the variate.

#### observe()

<a name="observe-1089"></a>

Observe the variate.

#### prune()

<a name="prune-1094"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-1090"></a>

Realize the variate.

#### register()

<a name="register-1105"></a>

Register with the diagnostic handler.

#### removeChild()

<a name="removechild-1100"></a>

Remove the child.

#### removeParent()

<a name="removeparent-1097"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-1107))

<a name="setchild-1099"></a>

Set the child.

#### setParent(u:[Delay](#delay-1107))

<a name="setparent-1096"></a>

Set the parent.

#### simulate()

<a name="simulate-1088"></a>

Simulate the variate.

#### trigger()

<a name="trigger-1106"></a>

Trigger an event with the diagnostic handler.


## DelayBoolean

<a name="delayboolean-177"></a>

  * Inherits from *[Delay](#delay-1107)*

Abstract delay variate with Boolean value.

| Assignment | Description |
| --- | --- |
| *[Boolean](#boolean-984)* | Value assignment. |
| *[String](#string-997)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Boolean](#boolean-984)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Boolean](#boolean-984)* | Value. |
| *w:[Real](#real-865)* | Weight. |


## DelayDiagnostics

<a name="delaydiagnostics-431"></a>

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
| *nodes:[Delay](#delay-1107)?\[\_\]* | Registered nodes. |
| *names:[String](#string-997)?\[\_\]* | Names of the nodes. |
| *xs:[Integer](#integer-668)?\[\_\]* | $x$-coordinates of the nodes. |
| *ys:[Integer](#integer-668)?\[\_\]* | $y$-coordinates of the nodes. |
| *n:[Integer](#integer-668)* | Number of nodes that have been registered. |
| *noutputs:[Integer](#integer-668)* | Number of graphs that have been output. |

| Member Function | Brief description |
| --- | --- |
| [register](#register-411) | Register a new node. |
| [name](#name-414) | Set the name of a node. |
| [position](#position-418) | Set the position of a node. |
| [trigger](#trigger-421) | Trigger an event. |
| [dot](#dot-430) | Output a dot graph of the current state. |


### Member Function Details

#### dot()

<a name="dot-430"></a>

Output a dot graph of the current state.

#### name(id:[Integer](#integer-668), name:[String](#string-997))

<a name="name-414"></a>

Set the name of a node.

  - id   : Id of the node.
  - name : The name.

#### position(id:[Integer](#integer-668), x:[Integer](#integer-668), y:[Integer](#integer-668))

<a name="position-418"></a>

Set the position of a node.

  - id : Id of the node.
  - x  : $x$-coordinate.
  - y  : $y$-coordinate.

#### register(o:[Delay](#delay-1107)) -> [Integer](#integer-668)

<a name="register-411"></a>

Register a new node. This is a callback function typically called
within the Delay class itself.

Returns an id assigned to the node.

#### trigger()

<a name="trigger-421"></a>

Trigger an event.


## DelayInteger

<a name="delayinteger-543"></a>

  * Inherits from *[Delay](#delay-1107)*

Abstract delay variate with Integer value.

| Assignment | Description |
| --- | --- |
| *[Integer](#integer-668)* | Value assignment. |
| *[String](#string-997)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Integer](#integer-668)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Integer](#integer-668)* | Value. |
| *w:[Real](#real-865)* | Weight. |


## DelayReal

<a name="delayreal-469"></a>

  * Inherits from *[Delay](#delay-1107)*

Abstract delay variate with Real value.

| Assignment | Description |
| --- | --- |
| *[Real](#real-865)* | Value assignment. |
| *[String](#string-997)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-865)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-865)* | Value. |
| *w:[Real](#real-865)* | Weight. |


## DelayRealVector

<a name="delayrealvector-776"></a>

  * Inherits from *[Delay](#delay-1107)*

Abstract delay variate with real vector value.

`D` Number of dimensions.

| Assignment | Description |
| --- | --- |
| *[Real](#real-865)\[\_\]* | Value assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-865)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-865)\[\_\]* | Value. |
| *w:[Real](#real-865)* | Weight. |


## FileOutputStream

<a name="fileoutputstream-480"></a>

  * Inherits from *[OutputStream](#outputstream-162)*

File output stream.

| Member Function | Brief description |
| --- | --- |
| [close](#close-479) | Close the file. |


### Member Function Details

#### close()

<a name="close-479"></a>

Close the file.


## Gamma

<a name="gamma-493"></a>

  * Inherits from *[DelayReal](#delayreal-469)*

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-865)* | Shape. |
| *θ:[Real](#real-865)* | Scale. |


## Gaussian

<a name="gaussian-266"></a>

  * Inherits from *[DelayReal](#delayreal-469)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-865)* | Mean. |
| *σ2:[Real](#real-865)* | Variance. |


## MultivariateGaussian

<a name="multivariategaussian-449"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-776)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-865)\[\_\]* | Mean. |
| *Σ:[Real](#real-865)\[\_,\_\]* | Covariance matrix. |


## OutputStream

<a name="outputstream-162"></a>

Output stream.

| Member Function | Brief description |
| --- | --- |
| [printf](#printf-130) | Print with format. |
| [printf](#printf-133) | Print with format. |
| [printf](#printf-136) | Print with format. |
| [printf](#printf-139) | Print with format. |
| [print](#print-141) | Print scalar. |
| [print](#print-143) | Print scalar. |
| [print](#print-145) | Print scalar. |
| [print](#print-147) | Print scalar. |
| [print](#print-150) | Print vector. |
| [print](#print-153) | Print vector. |
| [print](#print-157) | Print matrix. |
| [print](#print-161) | Print matrix. |


### Member Function Details

#### print(value:[Boolean](#boolean-984))

<a name="print-141"></a>

Print scalar.

#### print(value:[Integer](#integer-668))

<a name="print-143"></a>

Print scalar.

#### print(value:[Real](#real-865))

<a name="print-145"></a>

Print scalar.

#### print(value:[String](#string-997))

<a name="print-147"></a>

Print scalar.

#### print(x:[Integer](#integer-668)\[\_\])

<a name="print-150"></a>

Print vector.

#### print(x:[Real](#real-865)\[\_\])

<a name="print-153"></a>

Print vector.

#### print(X:[Integer](#integer-668)\[\_,\_\])

<a name="print-157"></a>

Print matrix.

#### print(X:[Real](#real-865)\[\_,\_\])

<a name="print-161"></a>

Print matrix.

#### printf(fmt:[String](#string-997), value:[Boolean](#boolean-984))

<a name="printf-130"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-997), value:[Integer](#integer-668))

<a name="printf-133"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-997), value:[Real](#real-865))

<a name="printf-136"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-997), value:[String](#string-997))

<a name="printf-139"></a>

Print with format. See system `printf`.


## StdErrStream

<a name="stderrstream-124"></a>

  * Inherits from *[OutputStream](#outputstream-162)*

Output stream for stderr.


## StdOutStream

<a name="stdoutstream-113"></a>

  * Inherits from *[OutputStream](#outputstream-162)*

Output stream for stdout.


## Uniform

<a name="uniform-247"></a>

  * Inherits from *[DelayReal](#delayreal-469)*

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-865)* | Lower bound. |
| *u:[Real](#real-865)* | Upper bound. |

