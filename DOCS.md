
# Summary

| Variable | Description |
| --- | --- |
| *delayDiagnostics:[DelayDiagnostics](#delaydiagnostics-527)?* | Global diagnostics handler for delayed sampling. |
| *inf:[Real64](#real64-777)* | $\infty$ |
| *stderr:[StdErrStream](#stderrstream-124)* | Standard error. |
| *stdout:[StdOutStream](#stdoutstream-113)* | Standard output. |
| *π:[Real64](#real64-777)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-399) | Create. |
| [Beta](#beta-613) | Create a Beta distribution. |
| [Boolean](#boolean-974) | Convert other basic types to Boolean. |
| [Gamma](#gamma-593) | Create Gamma distribution. |
| [Gaussian](#gaussian-386) | Create a Gaussian distribution. |
| [Gaussian](#gaussian-550) | Create. |
| [Integer](#integer-766) | Convert other basic types to Integer. |
| [Integer32](#integer32-901) | Convert other basic types to Integer32. |
| [Integer64](#integer64-402) | Convert other basic types to Integer64. |
| [Real](#real-963) | Convert other basic types to Real. |
| [Real32](#real32-41) | Convert other basic types to Real32. |
| [Real64](#real64-779) | Convert other basic types to Real64. |
| [String](#string-987) | Convert other basic types to String. |
| [Uniform](#uniform-367) | Create a Uniform distribution. |
| [abs](#abs-945) | Absolute value. |
| [abs](#abs-446) | Absolute value. |
| [abs](#abs-85) | Absolute value. |
| [abs](#abs-823) | Absolute value. |
| [adjacent_difference](#adjacent-difference-350) | Inclusive prefix sum. |
| [ancestor](#ancestor-11) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-5) | Sample an ancestry vector for a vector of log-weights. |
| [columns](#columns-184) | Number of columns of a matrix. |
| [columns](#columns-186) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-26) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-38) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-748) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-346) | Inclusive prefix sum. |
| [fclose](#fclose-573) | Close a file. |
| [fopen](#fopen-568) | Open a file for reading. |
| [fopen](#fopen-571) | Open a file. |
| [identity](#identity-201) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-342) | Inclusive prefix sum. |
| [inverse](#inverse-752) | Inverse of a matrix. |
| [isnan](#isnan-96) | Does this have the value NaN? |
| [isnan](#isnan-834) | Does this have the value NaN? |
| [length](#length-1050) | Length of a string. |
| [length](#length-615) | Length of a vector. |
| [length](#length-617) | Length of a vector. |
| [llt](#llt-755) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-338) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-195) | Create matrix filled with a given scalar. |
| [max](#max-951) | Maximum of two values. |
| [max](#max-452) | Maximum of two values. |
| [max](#max-91) | Maximum of two values. |
| [max](#max-829) | Maximum of two values. |
| [max](#max-329) | Maximum of a vector. |
| [min](#min-954) | Minimum of two values. |
| [min](#min-455) | Minimum of two values. |
| [min](#min-94) | Minimum of two values. |
| [min](#min-832) | Minimum of two values. |
| [min](#min-333) | Minimum of a vector. |
| [mod](#mod-948) | Modulus. |
| [mod](#mod-449) | Modulus. |
| [mod](#mod-88) | Modulus. |
| [mod](#mod-826) | Modulus. |
| [norm](#norm-744) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-32) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [read](#read-960) | Read numbers from a file. |
| [rows](#rows-180) | Number of rows of a matrix. |
| [rows](#rows-182) | Number of rows of a matrix. |
| [scalar](#scalar-188) | Convert single-element matrix to scalar. |
| [scalar](#scalar-619) | Convert single-element vector to scalar. |
| [seed](#seed-98) | Seed the pseudorandom number generator. |
| [simulate_bernoulli](#simulate-bernoulli-100) | Simulate a Bernoulli variate. |
| [simulate_binomial](#simulate-binomial-103) | Simulate a Binomial variate. |
| [simulate_gamma](#simulate-gamma-112) | Simulate a Gamma variate. |
| [simulate_gaussian](#simulate-gaussian-109) | Simulate a Gaussian variate. |
| [simulate_uniform](#simulate-uniform-106) | Simulate a Uniform variate. |
| [solve](#solve-759) | Solve a system of equations. |
| [solve](#solve-763) | Solve a system of equations. |
| [squaredNorm](#squarednorm-746) | Squared norm of a vector. |
| [sum](#sum-325) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-18) | Systematic resampling. |
| [transpose](#transpose-750) | Transpose of a matrix. |
| [vector](#vector-624) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-123) | Build the project. |
| [check](#check-321) | Check the file structure of the project for possible issues. |
| [clean](#clean-640) | Clean the project directory of all build files. |
| [dist](#dist-320) | Build a distributable archive for the project. |
| [docs](#docs-203) | Build the reference documentation for the project. |
| [init](#init-776) | Initialise the working directory for a new project. |
| [install](#install-202) | Install the project. |
| [uninstall](#uninstall-178) | Uninstall the project. |

| Basic Type | Brief description |
| --- | --- |
| [Boolean](#boolean-972) | A Boolean value. |
| [File](#file-566) | A file handle. |
| [Integer32](#integer32-899) | A 32-bit integer. |
| [Integer64](#integer64-400) | A 64-bit integer. |
| [Real32](#real32-39) | A 32-bit (single precision) floating point value. |
| [Real64](#real64-777) | A 64-bit (double precision) floating point value. |
| [String](#string-985) | A string value. |

| Alias Type | Brief description |
| --- | --- |
| [Integer](#integer-764) | An integer value of default type. |
| [Real](#real-961) | A floating point value of default type. |

| Class Type | Brief description |
| --- | --- |
| [AffineGaussian](#affinegaussian-850) | Gaussian that has a mean which is an affine transformation of another Gaussian. |
| [AffineGaussianExpression](#affinegaussianexpression-644) | Expression used to accumulate affine transformations of Gaussians. |
| [AffineMultivariateGaussian](#affinemultivariategaussian-890) | Multivariate Gaussian that has a mean which is an affine transformation of another multivariate Gaussian. |
| [AffineMultivariateGaussianExpression](#affinemultivariategaussianexpression-459) | Expression used to accumulate affine transformations of multivariate Gaussians. |
| [Bernoulli](#bernoulli-396) | Bernoulli distribution. |
| [Beta](#beta-609) | Beta distribution. |
| [Delay](#delay-1095) | Node interface for delayed sampling. |
| [DelayBoolean](#delayboolean-177) | Abstract delay variate with Boolean value. |
| [DelayDiagnostics](#delaydiagnostics-527) | Outputs graphical representations of the delayed sampling state for diagnostic purposes. |
| [DelayInteger](#delayinteger-639) | Abstract delay variate with Integer value. |
| [DelayReal](#delayreal-565) | Abstract delay variate with Real value. |
| [DelayRealVector](#delayrealvector-872) | Abstract delay variate with real vector value. |
| [FileOutputStream](#fileoutputstream-576) | File output stream. |
| [Gamma](#gamma-589) | Gamma distribution. |
| [Gaussian](#gaussian-382) | Gaussian distribution. |
| [MultivariateGaussian](#multivariategaussian-545) | Multivariate Gaussian distribution. |
| [OutputStream](#outputstream-162) | Output stream. |
| [StdErrStream](#stderrstream-124) | Output stream for stderr. |
| [StdOutStream](#stdoutstream-113) | Output stream for stdout. |
| [Uniform](#uniform-363) | Uniform distribution. |


# Function Details

#### Bernoulli(ρ:[Real](#real-961)) -> [Bernoulli](#bernoulli-396)

<a name="bernoulli-399"></a>

Create.

#### Beta(α:[Real](#real-961), β:[Real](#real-961)) -> [Beta](#beta-609)

<a name="beta-613"></a>

Create a Beta distribution.

#### Boolean(x:[Boolean](#boolean-972)) -> [Boolean](#boolean-972)

<a name="boolean-974"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-961), θ:[Real](#real-961)) -> [Gamma](#gamma-589)

<a name="gamma-593"></a>

Create Gamma distribution.

#### Gaussian(μ:[Real](#real-961), σ2:[Real](#real-961)) -> [Gaussian](#gaussian-382)

<a name="gaussian-386"></a>

Create a Gaussian distribution.

#### Gaussian(μ:[Real](#real-961)\[\_\], Σ:[Real](#real-961)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-545)

<a name="gaussian-550"></a>

Create.

#### Integer(x:[Integer64](#integer64-400)) -> [Integer](#integer-764)

<a name="integer-766"></a>

Convert other basic types to Integer. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer32(x:[Integer32](#integer32-899)) -> [Integer32](#integer32-899)

<a name="integer32-901"></a>

Convert other basic types to Integer32. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer64(x:[Integer64](#integer64-400)) -> [Integer64](#integer64-400)

<a name="integer64-402"></a>

Convert other basic types to Integer64. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real(x:[Real64](#real64-777)) -> [Real](#real-961)

<a name="real-963"></a>

Convert other basic types to Real. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real32(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="real32-41"></a>

Convert other basic types to Real32. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### Real64(x:[Real64](#real64-777)) -> [Real64](#real64-777)

<a name="real64-779"></a>

Convert other basic types to Real64. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### String(x:[String](#string-985)) -> [String](#string-985)

<a name="string-987"></a>

Convert other basic types to String. This is overloaded for Bolean, Real64,
String, Integer64, Integer32 and String.

#### Uniform(l:[Real](#real-961), u:[Real](#real-961)) -> [Uniform](#uniform-363)

<a name="uniform-367"></a>

Create a Uniform distribution.

#### abs(x:[Integer32](#integer32-899)) -> [Integer32](#integer32-899)

<a name="abs-945"></a>

Absolute value.

#### abs(x:[Integer64](#integer64-400)) -> [Integer64](#integer64-400)

<a name="abs-446"></a>

Absolute value.

#### abs(x:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="abs-85"></a>

Absolute value.

#### abs(x:[Real64](#real64-777)) -> [Real64](#real64-777)

<a name="abs-823"></a>

Absolute value.

#### adjacent_difference(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)\[\_\]

<a name="adjacent-difference-350"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-961)\[\_\]) -> [Integer](#integer-764)

<a name="ancestor-11"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-961)\[\_\]) -> [Integer](#integer-764)\[\_\]

<a name="ancestors-5"></a>

Sample an ancestry vector for a vector of log-weights.

#### columns(X:[Real](#real-961)\[\_,\_\]) -> [Integer64](#integer64-400)

<a name="columns-184"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-764)\[\_,\_\]) -> [Integer64](#integer64-400)

<a name="columns-186"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-764)\[\_\]) -> [Integer](#integer-764)\[\_\]

<a name="cumulative-offspring-to-ancestors-26"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-961)\[\_\]) -> [Real](#real-961)\[\_\]

<a name="cumulative-weights-38"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-961)\[\_,\_\]) -> [Real](#real-961)

<a name="determinant-748"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)\[\_\]

<a name="exclusive-prefix-sum-346"></a>

Inclusive prefix sum.

#### fclose(file:[File](#file-566))

<a name="fclose-573"></a>

Close a file.

#### fopen(file:[String](#string-985)) -> [File](#file-566)

<a name="fopen-568"></a>

Open a file for reading.

  - file : The file name.

#### fopen(file:[String](#string-985), mode:[String](#string-985)) -> [File](#file-566)

<a name="fopen-571"></a>

Open a file.

  - file : The file name.
  - mode : The mode, either `r` (read), `w` (write), `a` (append) or any
    other modes as in system `fopen`.

#### identity(rows:[Integer](#integer-764), columns:[Integer](#integer-764)) -> [Real](#real-961)\[\_,\_\]

<a name="identity-201"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)\[\_\]

<a name="inclusive-prefix-sum-342"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-961)\[\_,\_\]) -> [Real](#real-961)\[\_,\_\]

<a name="inverse-752"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-39)) -> [Boolean](#boolean-972)

<a name="isnan-96"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-777)) -> [Boolean](#boolean-972)

<a name="isnan-834"></a>

Does this have the value NaN?

#### length(x:[String](#string-985)) -> [Integer](#integer-764)

<a name="length-1050"></a>

Length of a string.

#### length(x:[Real](#real-961)\[\_\]) -> [Integer64](#integer64-400)

<a name="length-615"></a>

Length of a vector.

#### length(x:[Integer](#integer-764)\[\_\]) -> [Integer64](#integer64-400)

<a name="length-617"></a>

Length of a vector.

#### llt(X:[Real](#real-961)\[\_,\_\]) -> [Real](#real-961)\[\_,\_\]

<a name="llt-755"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)

<a name="log-sum-exp-338"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-961), rows:[Integer](#integer-764), columns:[Integer](#integer-764)) -> [Real](#real-961)\[\_,\_\]

<a name="matrix-195"></a>

Create matrix filled with a given scalar.

#### max(x:[Integer32](#integer32-899), y:[Integer32](#integer32-899)) -> [Integer32](#integer32-899)

<a name="max-951"></a>

Maximum of two values.

#### max(x:[Integer64](#integer64-400), y:[Integer64](#integer64-400)) -> [Integer64](#integer64-400)

<a name="max-452"></a>

Maximum of two values.

#### max(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="max-91"></a>

Maximum of two values.

#### max(x:[Real64](#real64-777), y:[Real64](#real64-777)) -> [Real64](#real64-777)

<a name="max-829"></a>

Maximum of two values.

#### max(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)

<a name="max-329"></a>

Maximum of a vector.

#### min(x:[Integer32](#integer32-899), y:[Integer32](#integer32-899)) -> [Integer32](#integer32-899)

<a name="min-954"></a>

Minimum of two values.

#### min(x:[Integer64](#integer64-400), y:[Integer64](#integer64-400)) -> [Integer64](#integer64-400)

<a name="min-455"></a>

Minimum of two values.

#### min(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="min-94"></a>

Minimum of two values.

#### min(x:[Real64](#real64-777), y:[Real64](#real64-777)) -> [Real64](#real64-777)

<a name="min-832"></a>

Minimum of two values.

#### min(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)

<a name="min-333"></a>

Minimum of a vector.

#### mod(x:[Integer32](#integer32-899), y:[Integer32](#integer32-899)) -> [Integer32](#integer32-899)

<a name="mod-948"></a>

Modulus.

#### mod(x:[Integer64](#integer64-400), y:[Integer64](#integer64-400)) -> [Integer64](#integer64-400)

<a name="mod-449"></a>

Modulus.

#### mod(x:[Real32](#real32-39), y:[Real32](#real32-39)) -> [Real32](#real32-39)

<a name="mod-88"></a>

Modulus.

#### mod(x:[Real64](#real64-777), y:[Real64](#real64-777)) -> [Real64](#real64-777)

<a name="mod-826"></a>

Modulus.

#### norm(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)

<a name="norm-744"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-764)\[\_\]) -> [Integer](#integer-764)\[\_\]

<a name="permute-ancestors-32"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### read(file:[String](#string-985), N:[Integer](#integer-764)) -> [Real](#real-961)\[\_\]

<a name="read-960"></a>

Read numbers from a file.

#### rows(X:[Real](#real-961)\[\_,\_\]) -> [Integer64](#integer64-400)

<a name="rows-180"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-764)\[\_,\_\]) -> [Integer64](#integer64-400)

<a name="rows-182"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-961)\[\_,\_\]) -> [Real](#real-961)

<a name="scalar-188"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)

<a name="scalar-619"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-764))

<a name="seed-98"></a>

Seed the pseudorandom number generator.

- seed: Seed value.

#### simulate_bernoulli(ρ:[Real](#real-961)) -> [Boolean](#boolean-972)

<a name="simulate-bernoulli-100"></a>

Simulate a Bernoulli variate.

- ρ: Probability of a true result.

#### simulate_binomial(n:[Integer](#integer-764), ρ:[Real](#real-961)) -> [Integer](#integer-764)

<a name="simulate-binomial-103"></a>

Simulate a Binomial variate.

- n: Number of trials.
- ρ: Probability of a true result.

#### simulate_gamma(k:[Real](#real-961), θ:[Real](#real-961)) -> [Real](#real-961)

<a name="simulate-gamma-112"></a>

Simulate a Gamma variate.

- k: Shape.
- θ: Scale.

#### simulate_gaussian(μ:[Real](#real-961), σ2:[Real](#real-961)) -> [Real](#real-961)

<a name="simulate-gaussian-109"></a>

Simulate a Gaussian variate.

- μ: Mean.
- σ2: Variance.

#### simulate_uniform(l:[Real](#real-961), u:[Real](#real-961)) -> [Real](#real-961)

<a name="simulate-uniform-106"></a>

Simulate a Uniform variate.

- l: Lower bound of interval.
- u: Upper bound of interval.

#### solve(X:[Real](#real-961)\[\_,\_\], y:[Real](#real-961)\[\_\]) -> [Real](#real-961)\[\_\]

<a name="solve-759"></a>

Solve a system of equations.

#### solve(X:[Real](#real-961)\[\_,\_\], Y:[Real](#real-961)\[\_,\_\]) -> [Real](#real-961)\[\_,\_\]

<a name="solve-763"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)

<a name="squarednorm-746"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-961)\[\_\]) -> [Real](#real-961)

<a name="sum-325"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-961)\[\_\]) -> [Integer](#integer-764)\[\_\]

<a name="systematic-cumulative-offspring-18"></a>

Systematic resampling.

#### transpose(X:[Real](#real-961)\[\_,\_\]) -> [Real](#real-961)\[\_,\_\]

<a name="transpose-750"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-961), length:[Integer](#integer-764)) -> [Real](#real-961)\[\_\]

<a name="vector-624"></a>

Create vector filled with a given scalar.


# Program Details

#### build(include_dir:[String](#string-985), lib_dir:[String](#string-985), share_dir:[String](#string-985), prefix:[String](#string-985), warnings:[Boolean](#boolean-972) <- true, debug:[Boolean](#boolean-972) <- true, verbose:[Boolean](#boolean-972) <- true)

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

<a name="check-321"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-640"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-320"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-203"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-985) <- "untitled")

<a name="init-776"></a>

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

<a name="boolean-972"></a>

A Boolean value.

#### type File

<a name="file-566"></a>

A file handle.

#### type Integer32

<a name="integer32-899"></a>

A 32-bit integer.

#### type Integer64

<a name="integer64-400"></a>

A 64-bit integer.

#### type Real32

<a name="real32-39"></a>

A 32-bit (single precision) floating point value.

#### type Real64

<a name="real64-777"></a>

A 64-bit (double precision) floating point value.

#### type String

<a name="string-985"></a>

A string value.


# Alias Type Details

#### type Integer = [Integer64](#integer64-400)

<a name="integer-764"></a>

An integer value of default type.

#### type Real = [Real64](#real64-777)

<a name="real-961"></a>

A floating point value of default type.


# Class Type Details


## AffineGaussian

<a name="affinegaussian-850"></a>

  * Inherits from *[Gaussian](#gaussian-382)*

Gaussian that has a mean which is an affine transformation of another
Gaussian.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-961)* | Multiplicative scalar of affine transformation. |
| *μ:[Gaussian](#gaussian-382)* | Mean. |
| *c:[Real](#real-961)* | Additive scalar of affine transformation. |
| *q:[Real](#real-961)* | Variance. |
| *y:[Real](#real-961)* | Marginalized prior mean. |
| *s:[Real](#real-961)* | Marginalized prior variance. |


## AffineGaussianExpression

<a name="affinegaussianexpression-644"></a>

Expression used to accumulate affine transformations of Gaussians.

  - `a` Multiplicative scalar of affine transformation.
  - `u` Parent.
  - `c` Additive scalar of affine transformation.

| Conversion | Description |
| --- | --- |
| *[Real](#real-961)* | Value conversion. |


## AffineMultivariateGaussian

<a name="affinemultivariategaussian-890"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-545)*

Multivariate Gaussian that has a mean which is an affine transformation of
another multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-961)\[\_,\_\]* | Matrix of affine transformation. |
| *μ:[MultivariateGaussian](#multivariategaussian-545)* | Mean. |
| *c:[Real](#real-961)\[\_\]* | Vector of affine transformation. |
| *Q:[Real](#real-961)\[\_,\_\]* | Disturbance covariance. |
| *y:[Real](#real-961)\[\_\]* | Marginalized prior mean. |
| *S:[Real](#real-961)\[\_,\_\]* | Marginalized prior covariance. |


## AffineMultivariateGaussianExpression

<a name="affinemultivariategaussianexpression-459"></a>

Expression used to accumulate affine transformations of multivariate
Gaussians.

  - `A` Matrix of affine transformation.
  - `u` Parent.
  - `c` Vector of affine transformation.

| Conversion | Description |
| --- | --- |
| *[Real](#real-961)\[\_\]* | Value conversion. |


## Bernoulli

<a name="bernoulli-396"></a>

  * Inherits from *[DelayBoolean](#delayboolean-177)*

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-961)* | Probability of a true result. |


## Beta

<a name="beta-609"></a>

  * Inherits from *[DelayReal](#delayreal-565)*

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-961)* | First shape parameter. |
| *β:[Real](#real-961)* | Second shape parameter. |


## Delay

<a name="delay-1095"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-764)* | State of the variate. |
| *missing:[Boolean](#boolean-972)* | Is the value missing? |
| *parent:[Delay](#delay-1095)?* | Parent. |
| *child:[Delay](#delay-1095)?* | Child, if one exists and it is on the stem. |
| *id:[Integer](#integer-764)* | Unique id for delayed sampling diagnostics. |
| *nforward:[Integer](#integer-764)* | Number of observations absorbed in forward pass, for delayed sampling diagnostics. |
| *nbackward:[Integer](#integer-764)* | Number of observations absorbed in backward pass, for delayed sampling diagnostics. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-1062) | Is this a root node? |
| [isTerminal](#isterminal-1063) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-1064) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-1065) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-1066) | Is this node in the marginalized state? |
| [isRealized](#isrealized-1067) | Is this node in the realized state? |
| [isMissing](#ismissing-1068) | Is the value of this node missing? |
| [isNotMissing](#isnotmissing-1069) | Is the value of this node not missing? |
| [initialize](#initialize-1070) | Initialize as a root node. |
| [initialize](#initialize-1072) | Initialize as a non-root node. |
| [absorb](#absorb-1074) | Increment number of observations absorbed. |
| [marginalize](#marginalize-1075) | Marginalize the variate. |
| [simulate](#simulate-1076) | Simulate the variate. |
| [observe](#observe-1077) | Observe the variate. |
| [realize](#realize-1078) | Realize the variate. |
| [graft](#graft-1079) | Graft the stem to this node. |
| [graft](#graft-1081) | Graft the stem to this node. |
| [prune](#prune-1082) | Prune the stem from below this node. |
| [setParent](#setparent-1084) | Set the parent. |
| [removeParent](#removeparent-1085) | Remove the parent. |
| [setChild](#setchild-1087) | Set the child. |
| [removeChild](#removechild-1088) | Remove the child. |
| [register](#register-1093) | Register with the diagnostic handler. |
| [trigger](#trigger-1094) | Trigger an event with the diagnostic handler. |


### Member Function Details

#### absorb(nbackward:[Integer](#integer-764))

<a name="absorb-1074"></a>

Increment number of observations absorbed.

  - `nbackward` : Number of new observations absorbed.

#### graft()

<a name="graft-1079"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-1095))

<a name="graft-1081"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-1070"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-1095))

<a name="initialize-1072"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-972)

<a name="isinitialized-1065"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-972)

<a name="ismarginalized-1066"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-972)

<a name="ismissing-1068"></a>

Is the value of this node missing?

#### isNotMissing() -> [Boolean](#boolean-972)

<a name="isnotmissing-1069"></a>

Is the value of this node not missing?

#### isRealized() -> [Boolean](#boolean-972)

<a name="isrealized-1067"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-972)

<a name="isroot-1062"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-972)

<a name="isterminal-1063"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-972)

<a name="isuninitialized-1064"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-1075"></a>

Marginalize the variate.

#### observe()

<a name="observe-1077"></a>

Observe the variate.

#### prune()

<a name="prune-1082"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-1078"></a>

Realize the variate.

#### register()

<a name="register-1093"></a>

Register with the diagnostic handler.

#### removeChild()

<a name="removechild-1088"></a>

Remove the child.

#### removeParent()

<a name="removeparent-1085"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-1095))

<a name="setchild-1087"></a>

Set the child.

#### setParent(u:[Delay](#delay-1095))

<a name="setparent-1084"></a>

Set the parent.

#### simulate()

<a name="simulate-1076"></a>

Simulate the variate.

#### trigger()

<a name="trigger-1094"></a>

Trigger an event with the diagnostic handler.


## DelayBoolean

<a name="delayboolean-177"></a>

  * Inherits from *[Delay](#delay-1095)*

Abstract delay variate with Boolean value.

| Assignment | Description |
| --- | --- |
| *[Boolean](#boolean-972)* | Value assignment. |
| *[String](#string-985)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Boolean](#boolean-972)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Boolean](#boolean-972)* | Value. |
| *w:[Real](#real-961)* | Weight. |


## DelayDiagnostics

<a name="delaydiagnostics-527"></a>

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
| *nodes:[Delay](#delay-1095)?\[\_\]* | Registered nodes. |
| *names:[String](#string-985)?\[\_\]* | Names of the nodes. |
| *xs:[Integer](#integer-764)?\[\_\]* | $x$-coordinates of the nodes. |
| *ys:[Integer](#integer-764)?\[\_\]* | $y$-coordinates of the nodes. |
| *n:[Integer](#integer-764)* | Number of nodes that have been registered. |
| *noutputs:[Integer](#integer-764)* | Number of graphs that have been output. |

| Member Function | Brief description |
| --- | --- |
| [register](#register-507) | Register a new node. |
| [name](#name-510) | Set the name of a node. |
| [position](#position-514) | Set the position of a node. |
| [trigger](#trigger-517) | Trigger an event. |
| [dot](#dot-526) | Output a dot graph of the current state. |


### Member Function Details

#### dot()

<a name="dot-526"></a>

Output a dot graph of the current state.

#### name(id:[Integer](#integer-764), name:[String](#string-985))

<a name="name-510"></a>

Set the name of a node.

  - id   : Id of the node.
  - name : The name.

#### position(id:[Integer](#integer-764), x:[Integer](#integer-764), y:[Integer](#integer-764))

<a name="position-514"></a>

Set the position of a node.

  - id : Id of the node.
  - x  : $x$-coordinate.
  - y  : $y$-coordinate.

#### register(o:[Delay](#delay-1095)) -> [Integer](#integer-764)

<a name="register-507"></a>

Register a new node. This is a callback function typically called
within the Delay class itself.

Returns an id assigned to the node.

#### trigger()

<a name="trigger-517"></a>

Trigger an event.


## DelayInteger

<a name="delayinteger-639"></a>

  * Inherits from *[Delay](#delay-1095)*

Abstract delay variate with Integer value.

| Assignment | Description |
| --- | --- |
| *[Integer](#integer-764)* | Value assignment. |
| *[String](#string-985)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Integer](#integer-764)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Integer](#integer-764)* | Value. |
| *w:[Real](#real-961)* | Weight. |


## DelayReal

<a name="delayreal-565"></a>

  * Inherits from *[Delay](#delay-1095)*

Abstract delay variate with Real value.

| Assignment | Description |
| --- | --- |
| *[Real](#real-961)* | Value assignment. |
| *[String](#string-985)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-961)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-961)* | Value. |
| *w:[Real](#real-961)* | Weight. |


## DelayRealVector

<a name="delayrealvector-872"></a>

  * Inherits from *[Delay](#delay-1095)*

Abstract delay variate with real vector value.

`D` Number of dimensions.

| Assignment | Description |
| --- | --- |
| *[Real](#real-961)\[\_\]* | Value assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-961)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-961)\[\_\]* | Value. |
| *w:[Real](#real-961)* | Weight. |


## FileOutputStream

<a name="fileoutputstream-576"></a>

  * Inherits from *[OutputStream](#outputstream-162)*

File output stream.

| Member Function | Brief description |
| --- | --- |
| [close](#close-575) | Close the file. |


### Member Function Details

#### close()

<a name="close-575"></a>

Close the file.


## Gamma

<a name="gamma-589"></a>

  * Inherits from *[DelayReal](#delayreal-565)*

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-961)* | Shape. |
| *θ:[Real](#real-961)* | Scale. |


## Gaussian

<a name="gaussian-382"></a>

  * Inherits from *[DelayReal](#delayreal-565)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-961)* | Mean. |
| *σ2:[Real](#real-961)* | Variance. |


## MultivariateGaussian

<a name="multivariategaussian-545"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-872)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-961)\[\_\]* | Mean. |
| *Σ:[Real](#real-961)\[\_,\_\]* | Covariance matrix. |


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

#### print(value:[Boolean](#boolean-972))

<a name="print-141"></a>

Print scalar.

#### print(value:[Integer](#integer-764))

<a name="print-143"></a>

Print scalar.

#### print(value:[Real](#real-961))

<a name="print-145"></a>

Print scalar.

#### print(value:[String](#string-985))

<a name="print-147"></a>

Print scalar.

#### print(x:[Integer](#integer-764)\[\_\])

<a name="print-150"></a>

Print vector.

#### print(x:[Real](#real-961)\[\_\])

<a name="print-153"></a>

Print vector.

#### print(X:[Integer](#integer-764)\[\_,\_\])

<a name="print-157"></a>

Print matrix.

#### print(X:[Real](#real-961)\[\_,\_\])

<a name="print-161"></a>

Print matrix.

#### printf(fmt:[String](#string-985), value:[Boolean](#boolean-972))

<a name="printf-130"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-985), value:[Integer](#integer-764))

<a name="printf-133"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-985), value:[Real](#real-961))

<a name="printf-136"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-985), value:[String](#string-985))

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

<a name="uniform-363"></a>

  * Inherits from *[DelayReal](#delayreal-565)*

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-961)* | Lower bound. |
| *u:[Real](#real-961)* | Upper bound. |

