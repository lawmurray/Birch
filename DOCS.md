
# Summary

| Variable | Description |
| --- | --- |
| *delayDiagnostics:[DelayDiagnostics](#delaydiagnostics-429)?* | Global diagnostics handler for delayed sampling. |
| *inf:[Real64](#real64-613)* | $\infty$ |
| *π:[Real64](#real64-613)* | $\pi$ |

| Function | Brief description |
| --- | --- |
| [Bernoulli](#bernoulli-232) | Create. |
| [Beta](#beta-909) | Create. |
| [Boolean](#boolean-758) | Convert other basic types to Boolean. |
| [Gamma](#gamma-321) | Create. |
| [Gaussian](#gaussian-311) | Create. |
| [Gaussian](#gaussian-449) | Create. |
| [Integer](#integer-771) | Convert other basic types to Integer. |
| [Integer32](#integer32-513) | Convert other basic types to Integer32. |
| [Integer64](#integer64-705) | Convert other basic types to Integer64. |
| [Real](#real-694) | Convert other basic types to Real. |
| [Real32](#real32-243) | Convert other basic types to Real32. |
| [Real64](#real64-615) | Convert other basic types to Real64. |
| [String](#string-792) | Convert other basic types to String. |
| [Uniform](#uniform-789) | Create. |
| [abs](#abs-557) | Absolute value. |
| [abs](#abs-749) | Absolute value. |
| [abs](#abs-287) | Absolute value. |
| [abs](#abs-659) | Absolute value. |
| [adjacent_difference](#adjacent-difference-37) | Inclusive prefix sum. |
| [ancestor](#ancestor-332) | Sample a single ancestor for a vector of log-weights. |
| [ancestors](#ancestors-326) | Sample an ancestry vector for a vector of log-weights. |
| [columns](#columns-55) | Number of columns of a matrix. |
| [columns](#columns-57) | Number of columns of a matrix. |
| [cumulative_offspring_to_ancestors](#cumulative-offspring-to-ancestors-347) | Convert a cumulative offspring vector into an ancestry vector. |
| [cumulative_weights](#cumulative-weights-359) | Compute the cumulative weight vector from the log-weight vector. |
| [determinant](#determinant-881) | Determinant of a matrix. |
| [exclusive_prefix_sum](#exclusive-prefix-sum-33) | Inclusive prefix sum. |
| [identity](#identity-72) | Create identity matrix. |
| [inclusive_prefix_sum](#inclusive-prefix-sum-29) | Inclusive prefix sum. |
| [inverse](#inverse-885) | Inverse of a matrix. |
| [isnan](#isnan-295) | Does this have the value NaN? |
| [isnan](#isnan-667) | Does this have the value NaN? |
| [length](#length-825) | Length of a string. |
| [length](#length-39) | Length of a vector. |
| [length](#length-41) | Length of a vector. |
| [llt](#llt-888) | `LL^T` Cholesky decomposition of a matrix. |
| [log_sum_exp](#log-sum-exp-25) | Exponentiate and sum a vector, return the logarithm of the sum. |
| [matrix](#matrix-66) | Create matrix filled with a given scalar. |
| [max](#max-560) | Maximum of two values. |
| [max](#max-752) | Maximum of two values. |
| [max](#max-290) | Maximum of two values. |
| [max](#max-662) | Maximum of two values. |
| [max](#max-16) | Maximum of a vector. |
| [min](#min-563) | Minimum of two values. |
| [min](#min-755) | Minimum of two values. |
| [min](#min-293) | Minimum of two values. |
| [min](#min-665) | Minimum of two values. |
| [min](#min-20) | Minimum of a vector. |
| [norm](#norm-877) | Norm of a vector. |
| [permute_ancestors](#permute-ancestors-353) | Permute an ancestry vector to ensure that, when a particle survives, at least one of its instances remains in the same place. |
| [print](#print-204) | Print scalar. |
| [print](#print-206) | Print scalar. |
| [print](#print-208) | Print scalar. |
| [print](#print-210) | Print scalar. |
| [print](#print-213) | Print vector. |
| [print](#print-216) | Print vector. |
| [print](#print-220) | Print matrix. |
| [print](#print-224) | Print matrix. |
| [printf](#printf-193) | Print with format. |
| [printf](#printf-196) | Print with format. |
| [printf](#printf-199) | Print with format. |
| [printf](#printf-202) | Print with format. |
| [read](#read-6) | Read numbers from a file. |
| [rows](#rows-51) | Number of rows of a matrix. |
| [rows](#rows-53) | Number of rows of a matrix. |
| [scalar](#scalar-59) | Convert single-element matrix to scalar. |
| [scalar](#scalar-43) | Convert single-element vector to scalar. |
| [seed](#seed-8) | Seed the pseudorandom number generator. |
| [solve](#solve-892) | Solve a system of equations. |
| [solve](#solve-896) | Solve a system of equations. |
| [squaredNorm](#squarednorm-879) | Squared norm of a vector. |
| [sum](#sum-12) | Sum of a vector. |
| [systematic_cumulative_offspring](#systematic-cumulative-offspring-339) | Systematic resampling. |
| [transpose](#transpose-883) | Transpose of a matrix. |
| [vector](#vector-48) | Create vector filled with a given scalar. |

| Program | Brief description |
| --- | --- |
| [build](#build-240) | Build the project. |
| [check](#check-190) | Check the file structure of the project for possible issues. |
| [clean](#clean-564) | Clean the project directory of all build files. |
| [dist](#dist-189) | Build a distributable archive for the project. |
| [docs](#docs-74) | Build the reference documentation for the project. |
| [init](#init-612) | Initialise the working directory for a new project. |
| [install](#install-73) | Install the project. |
| [uninstall](#uninstall-49) | Uninstall the project. |

| Basic Type | Brief description |
| --- | --- |
| [Boolean](#boolean-756) | A Boolean value. |
| [Integer32](#integer32-511) | A 32-bit integer. |
| [Integer64](#integer64-703) | A 64-bit integer. |
| [Real32](#real32-241) | A 32-bit (single precision) floating point value. |
| [Real64](#real64-613) | A 64-bit (double precision) floating point value. |
| [String](#string-790) | A string value. |

| Alias Type | Brief description |
| --- | --- |
| [Integer](#integer-769) | An integer value of default type. |
| [Real](#real-692) | A floating point value of default type. |

| Class Type | Brief description |
| --- | --- |
| [AffineGaussian](#affinegaussian-683) | Gaussian that has a mean which is an affine transformation of another Gaussian. |
| [AffineGaussianExpression](#affinegaussianexpression-572) | Expression used to accumulate affine transformations of Gaussians. |
| [AffineMultivariateGaussian](#affinemultivariategaussian-485) | Multivariate Gaussian that has a mean which is an affine transformation of another multivariate Gaussian. |
| [AffineMultivariateGaussianExpression](#affinemultivariategaussianexpression-369) | Expression used to accumulate affine transformations of multivariate Gaussians. |
| [Bernoulli](#bernoulli-229) | Bernoulli distribution. |
| [Beta](#beta-905) | Beta distribution. |
| [Delay](#delay-948) | Node interface for delayed sampling. |
| [DelayDiagnostics](#delaydiagnostics-429) | Outputs graphical representations of the delayed sampling state for diagnostic purposes. |
| [DelayReal](#delayreal-467) | Abstract delay variate with real value. |
| [DelayRealVector](#delayrealvector-510) | Abstract delay variate with real vector value. |
| [Gamma](#gamma-317) | Gamma distribution. |
| [Gaussian](#gaussian-307) | Gaussian distribution. |
| [MultivariateGaussian](#multivariategaussian-444) | Multivariate Gaussian distribution. |
| [Uniform](#uniform-785) | Uniform distribution. |


# Function Details

#### Bernoulli(ρ:[Real](#real-692)) -> [Bernoulli](#bernoulli-229)

<a name="bernoulli-232"></a>

Create.

#### Beta(α:[Real](#real-692), β:[Real](#real-692)) -> [Beta](#beta-905)

<a name="beta-909"></a>

Create.

#### Boolean(x:[Boolean](#boolean-756)) -> [Boolean](#boolean-756)

<a name="boolean-758"></a>

Convert other basic types to Boolean. This is overloaded for Boolean and
String.

#### Gamma(k:[Real](#real-692), θ:[Real](#real-692)) -> [Gamma](#gamma-317)

<a name="gamma-321"></a>

Create.

#### Gaussian(μ:[Real](#real-692), σ2:[Real](#real-692)) -> [Gaussian](#gaussian-307)

<a name="gaussian-311"></a>

Create.

#### Gaussian(μ:[Real](#real-692)\[\_\], Σ:[Real](#real-692)\[\_,\_\]) -> [MultivariateGaussian](#multivariategaussian-444)

<a name="gaussian-449"></a>

Create.

#### Integer(x:[Integer64](#integer64-703)) -> [Integer](#integer-769)

<a name="integer-771"></a>

Convert other basic types to Integer. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer32(x:[Integer32](#integer32-511)) -> [Integer32](#integer32-511)

<a name="integer32-513"></a>

Convert other basic types to Integer32. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Integer64(x:[Integer64](#integer64-703)) -> [Integer64](#integer64-703)

<a name="integer64-705"></a>

Convert other basic types to Integer64. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real(x:[Real64](#real64-613)) -> [Real](#real-692)

<a name="real-694"></a>

Convert other basic types to Real. This is overloaded for Real64,
Real32, Integer64, Integer32 and String.

#### Real32(x:[Real32](#real32-241)) -> [Real32](#real32-241)

<a name="real32-243"></a>

Convert other basic types to Real32. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### Real64(x:[Real64](#real64-613)) -> [Real64](#real64-613)

<a name="real64-615"></a>

Convert other basic types to Real64. This is overloaded for Real64, Real32,
Integer64, Integer32 and String.

#### String(x:[String](#string-790)) -> [String](#string-790)

<a name="string-792"></a>

Convert other basic types to String. This is overloaded for Bolean, Real64,
String, Integer64, Integer32 and String.

#### Uniform(l:[Real](#real-692), u:[Real](#real-692)) -> [Uniform](#uniform-785)

<a name="uniform-789"></a>

Create.

#### abs(x:[Integer32](#integer32-511)) -> [Integer32](#integer32-511)

<a name="abs-557"></a>

Absolute value.

#### abs(x:[Integer64](#integer64-703)) -> [Integer64](#integer64-703)

<a name="abs-749"></a>

Absolute value.

#### abs(x:[Real32](#real32-241)) -> [Real32](#real32-241)

<a name="abs-287"></a>

Absolute value.

#### abs(x:[Real64](#real64-613)) -> [Real64](#real64-613)

<a name="abs-659"></a>

Absolute value.

#### adjacent_difference(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)\[\_\]

<a name="adjacent-difference-37"></a>

Inclusive prefix sum.

#### ancestor(w:[Real](#real-692)\[\_\]) -> [Integer](#integer-769)

<a name="ancestor-332"></a>

Sample a single ancestor for a vector of log-weights.

#### ancestors(w:[Real](#real-692)\[\_\]) -> [Integer](#integer-769)\[\_\]

<a name="ancestors-326"></a>

Sample an ancestry vector for a vector of log-weights.

#### columns(X:[Real](#real-692)\[\_,\_\]) -> [Integer64](#integer64-703)

<a name="columns-55"></a>

Number of columns of a matrix.

#### columns(X:[Integer](#integer-769)\[\_,\_\]) -> [Integer64](#integer64-703)

<a name="columns-57"></a>

Number of columns of a matrix.

#### cumulative_offspring_to_ancestors(O:[Integer](#integer-769)\[\_\]) -> [Integer](#integer-769)\[\_\]

<a name="cumulative-offspring-to-ancestors-347"></a>

Convert a cumulative offspring vector into an ancestry vector.

#### cumulative_weights(w:[Real](#real-692)\[\_\]) -> [Real](#real-692)\[\_\]

<a name="cumulative-weights-359"></a>

Compute the cumulative weight vector from the log-weight vector.

#### determinant(X:[Real](#real-692)\[\_,\_\]) -> [Real](#real-692)

<a name="determinant-881"></a>

Determinant of a matrix.

#### exclusive_prefix_sum(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)\[\_\]

<a name="exclusive-prefix-sum-33"></a>

Inclusive prefix sum.

#### identity(rows:[Integer](#integer-769), columns:[Integer](#integer-769)) -> [Real](#real-692)\[\_,\_\]

<a name="identity-72"></a>

Create identity matrix.

#### inclusive_prefix_sum(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)\[\_\]

<a name="inclusive-prefix-sum-29"></a>

Inclusive prefix sum.

#### inverse(X:[Real](#real-692)\[\_,\_\]) -> [Real](#real-692)\[\_,\_\]

<a name="inverse-885"></a>

Inverse of a matrix.

#### isnan(x:[Real32](#real32-241)) -> [Boolean](#boolean-756)

<a name="isnan-295"></a>

Does this have the value NaN?

#### isnan(x:[Real64](#real64-613)) -> [Boolean](#boolean-756)

<a name="isnan-667"></a>

Does this have the value NaN?

#### length(x:[String](#string-790)) -> [Integer](#integer-769)

<a name="length-825"></a>

Length of a string.

#### length(x:[Real](#real-692)\[\_\]) -> [Integer64](#integer64-703)

<a name="length-39"></a>

Length of a vector.

#### length(x:[Integer](#integer-769)\[\_\]) -> [Integer64](#integer64-703)

<a name="length-41"></a>

Length of a vector.

#### llt(X:[Real](#real-692)\[\_,\_\]) -> [Real](#real-692)\[\_,\_\]

<a name="llt-888"></a>

`LL^T` Cholesky decomposition of a matrix.

#### log_sum_exp(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)

<a name="log-sum-exp-25"></a>

Exponentiate and sum a vector, return the logarithm of the sum.

#### matrix(x:[Real](#real-692), rows:[Integer](#integer-769), columns:[Integer](#integer-769)) -> [Real](#real-692)\[\_,\_\]

<a name="matrix-66"></a>

Create matrix filled with a given scalar.

#### max(x:[Integer32](#integer32-511), y:[Integer32](#integer32-511)) -> [Integer32](#integer32-511)

<a name="max-560"></a>

Maximum of two values.

#### max(x:[Integer64](#integer64-703), y:[Integer64](#integer64-703)) -> [Integer64](#integer64-703)

<a name="max-752"></a>

Maximum of two values.

#### max(x:[Real32](#real32-241), y:[Real32](#real32-241)) -> [Real32](#real32-241)

<a name="max-290"></a>

Maximum of two values.

#### max(x:[Real64](#real64-613), y:[Real64](#real64-613)) -> [Real64](#real64-613)

<a name="max-662"></a>

Maximum of two values.

#### max(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)

<a name="max-16"></a>

Maximum of a vector.

#### min(x:[Integer32](#integer32-511), y:[Integer32](#integer32-511)) -> [Integer32](#integer32-511)

<a name="min-563"></a>

Minimum of two values.

#### min(x:[Integer64](#integer64-703), y:[Integer64](#integer64-703)) -> [Integer64](#integer64-703)

<a name="min-755"></a>

Minimum of two values.

#### min(x:[Real32](#real32-241), y:[Real32](#real32-241)) -> [Real32](#real32-241)

<a name="min-293"></a>

Minimum of two values.

#### min(x:[Real64](#real64-613), y:[Real64](#real64-613)) -> [Real64](#real64-613)

<a name="min-665"></a>

Minimum of two values.

#### min(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)

<a name="min-20"></a>

Minimum of a vector.

#### norm(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)

<a name="norm-877"></a>

Norm of a vector.

#### permute_ancestors(a:[Integer](#integer-769)\[\_\]) -> [Integer](#integer-769)\[\_\]

<a name="permute-ancestors-353"></a>

Permute an ancestry vector to ensure that, when a particle survives, at
least one of its instances remains in the same place.

#### print(value:[Boolean](#boolean-756))

<a name="print-204"></a>

Print scalar.

#### print(value:[Integer](#integer-769))

<a name="print-206"></a>

Print scalar.

#### print(value:[Real](#real-692))

<a name="print-208"></a>

Print scalar.

#### print(value:[String](#string-790))

<a name="print-210"></a>

Print scalar.

#### print(x:[Integer](#integer-769)\[\_\])

<a name="print-213"></a>

Print vector.

#### print(x:[Real](#real-692)\[\_\])

<a name="print-216"></a>

Print vector.

#### print(X:[Integer](#integer-769)\[\_,\_\])

<a name="print-220"></a>

Print matrix.

#### print(X:[Real](#real-692)\[\_,\_\])

<a name="print-224"></a>

Print matrix.

#### printf(fmt:[String](#string-790), value:[Boolean](#boolean-756))

<a name="printf-193"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-790), value:[Integer](#integer-769))

<a name="printf-196"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-790), value:[Real](#real-692))

<a name="printf-199"></a>

Print with format. See system `printf`.

#### printf(fmt:[String](#string-790), value:[String](#string-790))

<a name="printf-202"></a>

Print with format. See system `printf`.

#### read(file:[String](#string-790), N:[Integer](#integer-769)) -> [Real](#real-692)\[\_\]

<a name="read-6"></a>

Read numbers from a file.

#### rows(X:[Real](#real-692)\[\_,\_\]) -> [Integer64](#integer64-703)

<a name="rows-51"></a>

Number of rows of a matrix.

#### rows(X:[Integer](#integer-769)\[\_,\_\]) -> [Integer64](#integer64-703)

<a name="rows-53"></a>

Number of rows of a matrix.

#### scalar(X:[Real](#real-692)\[\_,\_\]) -> [Real](#real-692)

<a name="scalar-59"></a>

Convert single-element matrix to scalar.

#### scalar(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)

<a name="scalar-43"></a>

Convert single-element vector to scalar.

#### seed(s:[Integer](#integer-769))

<a name="seed-8"></a>

Seed the pseudorandom number generator.

`seed` Seed.

#### solve(X:[Real](#real-692)\[\_,\_\], y:[Real](#real-692)\[\_\]) -> [Real](#real-692)\[\_\]

<a name="solve-892"></a>

Solve a system of equations.

#### solve(X:[Real](#real-692)\[\_,\_\], Y:[Real](#real-692)\[\_,\_\]) -> [Real](#real-692)\[\_,\_\]

<a name="solve-896"></a>

Solve a system of equations.

#### squaredNorm(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)

<a name="squarednorm-879"></a>

Squared norm of a vector.

#### sum(x:[Real](#real-692)\[\_\]) -> [Real](#real-692)

<a name="sum-12"></a>

Sum of a vector.

#### systematic_cumulative_offspring(W:[Real](#real-692)\[\_\]) -> [Integer](#integer-769)\[\_\]

<a name="systematic-cumulative-offspring-339"></a>

Systematic resampling.

#### transpose(X:[Real](#real-692)\[\_,\_\]) -> [Real](#real-692)\[\_,\_\]

<a name="transpose-883"></a>

Transpose of a matrix.

#### vector(x:[Real](#real-692), length:[Integer](#integer-769)) -> [Real](#real-692)\[\_\]

<a name="vector-48"></a>

Create vector filled with a given scalar.


# Program Details

#### build(include_dir:[String](#string-790), lib_dir:[String](#string-790), share_dir:[String](#string-790), prefix:[String](#string-790), warnings:[Boolean](#boolean-756) <- true, debug:[Boolean](#boolean-756) <- true, verbose:[Boolean](#boolean-756) <- true)

<a name="build-240"></a>

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

<a name="check-190"></a>

Check the file structure of the project for possible issues. This makes no
modifications to the project, but will output warnings for possible issues
such as:

  - files listed in the `MANIFEST` file that do not exist,
  - files of recognisable types that exist but that are not listed in the
    `MANIFEST` file, and
  - standard project meta files that do not exist.

#### clean()

<a name="clean-564"></a>

Clean the project directory of all build files.

#### dist()

<a name="dist-189"></a>

Build a distributable archive for the project. This creates an archive file
of the name `Example-x.y.z.tar.gz` in the working directory, where
`Example` is the name of the project and `x.y.z` the current version
number, as given in the `README.md` file.

#### docs()

<a name="docs-74"></a>

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory. It will be overwritten if
it already exists.

#### init(name:[String](#string-790) <- "untitled")

<a name="init-612"></a>

Initialise the working directory for a new project.

  - `--name` : Name of the project (default `untitled`).

#### install()

<a name="install-73"></a>

Install the project. This installs all header, library and data files
needed by the project into the directory specified by `--prefix` (or the
system default if this was not specified).

#### uninstall()

<a name="uninstall-49"></a>

Uninstall the project. This uninstalls all header, library and data files
from the directory specified by `--prefix` (or the system default if this
was not specified).


# Basic Type Details

#### type Boolean

<a name="boolean-756"></a>

A Boolean value.

#### type Integer32

<a name="integer32-511"></a>

A 32-bit integer.

#### type Integer64

<a name="integer64-703"></a>

A 64-bit integer.

#### type Real32

<a name="real32-241"></a>

A 32-bit (single precision) floating point value.

#### type Real64

<a name="real64-613"></a>

A 64-bit (double precision) floating point value.

#### type String

<a name="string-790"></a>

A string value.


# Alias Type Details

#### type Integer = [Integer64](#integer64-703)

<a name="integer-769"></a>

An integer value of default type.

#### type Real = [Real64](#real64-613)

<a name="real-692"></a>

A floating point value of default type.


# Class Type Details


## AffineGaussian

<a name="affinegaussian-683"></a>

  * Inherits from *[Gaussian](#gaussian-307)*

Gaussian that has a mean which is an affine transformation of another
Gaussian.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-692)* | Multiplicative scalar of affine transformation. |
| *μ:[Gaussian](#gaussian-307)* | Mean. |
| *c:[Real](#real-692)* | Additive scalar of affine transformation. |
| *q:[Real](#real-692)* | Variance. |
| *y:[Real](#real-692)* | Marginalized prior mean. |
| *s:[Real](#real-692)* | Marginalized prior variance. |


## AffineGaussianExpression

<a name="affinegaussianexpression-572"></a>

Expression used to accumulate affine transformations of Gaussians.

| Member Variable | Description |
| --- | --- |
| *a:[Real](#real-692)* | Multiplicative scalar of affine transformation. |
| *u:[Gaussian](#gaussian-307)* | Parent. |
| *c:[Real](#real-692)* | Additive scalar of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-571) | Initialize. |


### Member Function Details

#### initialize(a:[Real](#real-692), u:[Gaussian](#gaussian-307), c:[Real](#real-692))

<a name="initialize-571"></a>

Initialize.


## AffineMultivariateGaussian

<a name="affinemultivariategaussian-485"></a>

  * Inherits from *[MultivariateGaussian](#multivariategaussian-444)*

Multivariate Gaussian that has a mean which is an affine transformation of
another multivariate Gaussian.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-692)\[\_,\_\]* | Matrix of affine transformation. |
| *μ:[MultivariateGaussian](#multivariategaussian-444)* | Mean. |
| *c:[Real](#real-692)\[\_\]* | Vector of affine transformation. |
| *Q:[Real](#real-692)\[\_,\_\]* | Disturbance covariance. |
| *y:[Real](#real-692)\[\_\]* | Marginalized prior mean. |
| *S:[Real](#real-692)\[\_,\_\]* | Marginalized prior covariance. |


## AffineMultivariateGaussianExpression

<a name="affinemultivariategaussianexpression-369"></a>

Expression used to accumulate affine transformations of multivariate
Gaussians.

`R` Number of rows in transformation.
`C` Number of columns in transformation.

| Member Variable | Description |
| --- | --- |
| *A:[Real](#real-692)\[\_,\_\]* | Matrix of affine transformation. |
| *u:[MultivariateGaussian](#multivariategaussian-444)* | Parent. |
| *c:[Real](#real-692)\[\_\]* | Vector of affine transformation. |

| Member Function | Brief description |
| --- | --- |
| [initialize](#initialize-368) | Initialize. |


### Member Function Details

#### initialize(A:[Real](#real-692)\[\_,\_\], u:[MultivariateGaussian](#multivariategaussian-444), c:[Real](#real-692)\[\_\])

<a name="initialize-368"></a>

Initialize.


## Bernoulli

<a name="bernoulli-229"></a>

Bernoulli distribution.

| Member Variable | Description |
| --- | --- |
| *ρ:[Real](#real-692)* | Probability of a true result. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-226) | Simulate. |
| [observe](#observe-228) | Observe. |


### Member Function Details

#### observe(x:[Boolean](#boolean-756)) -> [Real](#real-692)

<a name="observe-228"></a>

Observe.

#### simulate() -> [Boolean](#boolean-756)

<a name="simulate-226"></a>

Simulate.


## Beta

<a name="beta-905"></a>

Beta distribution.

| Member Variable | Description |
| --- | --- |
| *α:[Real](#real-692)* | First shape parameter. |
| *β:[Real](#real-692)* | Second shape parameter. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-901) | Simulate. |
| [observe](#observe-904) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-692)) -> [Real](#real-692)

<a name="observe-904"></a>

Observe.

#### simulate() -> [Real](#real-692)

<a name="simulate-901"></a>

Simulate.


## Delay

<a name="delay-948"></a>

Node interface for delayed sampling.

| Member Variable | Description |
| --- | --- |
| *state:[Integer](#integer-769)* | State of the variate. |
| *missing:[Boolean](#boolean-756)* | Is the value missing? |
| *parent:[Delay](#delay-948)?* | Parent. |
| *child:[Delay](#delay-948)?* | Child, if one exists and it is on the stem. |
| *id:[Integer](#integer-769)* | Unique id for delayed sampling diagnostics. |

| Member Function | Brief description |
| --- | --- |
| [isRoot](#isroot-919) | Is this a root node? |
| [isTerminal](#isterminal-920) | Is this the terminal node of a stem? |
| [isUninitialized](#isuninitialized-921) | Is this node in the uninitialized state? |
| [isInitialized](#isinitialized-922) | Is this node in the initialized state? |
| [isMarginalized](#ismarginalized-923) | Is this node in the marginalized state? |
| [isRealized](#isrealized-924) | Is this node in the realized state? |
| [isMissing](#ismissing-925) | Is the value of this node missing? |
| [initialize](#initialize-926) | Initialize as a root node. |
| [initialize](#initialize-928) | Initialize as a non-root node. |
| [marginalize](#marginalize-929) | Marginalize the variate. |
| [forward](#forward-930) | Forward simulate the variate. |
| [realize](#realize-931) | Realize the variate. |
| [graft](#graft-932) | Graft the stem to this node. |
| [graft](#graft-934) | Graft the stem to this node. |
| [prune](#prune-935) | Prune the stem from below this node. |
| [setParent](#setparent-937) | Set the parent. |
| [removeParent](#removeparent-938) | Remove the parent. |
| [setChild](#setchild-940) | Set the child. |
| [removeChild](#removechild-941) | Remove the child. |
| [register](#register-946) | Register with the diagnostic handler. |
| [trigger](#trigger-947) | Trigger an event with the diagnostic handler. |


### Member Function Details

#### forward()

<a name="forward-930"></a>

Forward simulate the variate.

#### graft()

<a name="graft-932"></a>

Graft the stem to this node.

#### graft(c:[Delay](#delay-948))

<a name="graft-934"></a>

Graft the stem to this node.

`c` The child node that called this, and that will itself be part
of the stem.

#### initialize()

<a name="initialize-926"></a>

Initialize as a root node.

#### initialize(parent:[Delay](#delay-948))

<a name="initialize-928"></a>

Initialize as a non-root node.

`parent` The parent node.

#### isInitialized() -> [Boolean](#boolean-756)

<a name="isinitialized-922"></a>

Is this node in the initialized state?

#### isMarginalized() -> [Boolean](#boolean-756)

<a name="ismarginalized-923"></a>

Is this node in the marginalized state?

#### isMissing() -> [Boolean](#boolean-756)

<a name="ismissing-925"></a>

Is the value of this node missing?

#### isRealized() -> [Boolean](#boolean-756)

<a name="isrealized-924"></a>

Is this node in the realized state?

#### isRoot() -> [Boolean](#boolean-756)

<a name="isroot-919"></a>

Is this a root node?

#### isTerminal() -> [Boolean](#boolean-756)

<a name="isterminal-920"></a>

Is this the terminal node of a stem?

#### isUninitialized() -> [Boolean](#boolean-756)

<a name="isuninitialized-921"></a>

Is this node in the uninitialized state?

#### marginalize()

<a name="marginalize-929"></a>

Marginalize the variate.

#### prune()

<a name="prune-935"></a>

Prune the stem from below this node.

#### realize()

<a name="realize-931"></a>

Realize the variate.

#### register()

<a name="register-946"></a>

Register with the diagnostic handler.

#### removeChild()

<a name="removechild-941"></a>

Remove the child.

#### removeParent()

<a name="removeparent-938"></a>

Remove the parent.

#### setChild(u:[Delay](#delay-948))

<a name="setchild-940"></a>

Set the child.

#### setParent(u:[Delay](#delay-948))

<a name="setparent-937"></a>

Set the parent.

#### trigger()

<a name="trigger-947"></a>

Trigger an event with the diagnostic handler.


## DelayDiagnostics

<a name="delaydiagnostics-429"></a>

Outputs graphical representations of the delayed sampling state for
diagnostic purposes.

  - N : maximum number of variates

| Member Variable | Description |
| --- | --- |
| *nodes:[Delay](#delay-948)?\[\_\]* | Registered variates. |
| *names:[String](#string-790)\[\_\]* | Names associated with the variates. |
| *xs:[Integer](#integer-769)\[\_\]* | $x$-coordinates of the variates. |
| *ys:[Integer](#integer-769)\[\_\]* | $y$-coordinates of the variates. |
| *n:[Integer](#integer-769)* | Number of variates that have been registered. |
| *events:[Integer](#integer-769)* | Number of events that have been triggered. |

| Member Function | Brief description |
| --- | --- |
| [register](#register-419) | Register a new variate. |
| [name](#name-422) | Set the name of a previously-registered node. |
| [position](#position-426) | Set the position of a previously-registered node. |
| [trigger](#trigger-427) | Trigger an event. |
| [dot](#dot-428) | Draw a graph, in the dot language, for the current state. |


### Member Function Details

#### dot()

<a name="dot-428"></a>

Draw a graph, in the dot language, for the current state. The graph is
put in a file `diagnostics/state*n*.dot`, where *n* is the current
event number.

#### name(id:[Integer](#integer-769), name:[String](#string-790))

<a name="name-422"></a>

Set the name of a previously-registered node.

  - id   : Id of the node.
  - name : The name.

#### position(id:[Integer](#integer-769), x:[Integer](#integer-769), y:[Integer](#integer-769))

<a name="position-426"></a>

Set the position of a previously-registered node.

  - id : Id of the node.
  - x  : $x$-coordinate.
  - y  : $y$-coordinate.

A zero $x$ or $y$ coordinate suggests that an automatic layout should be
used for this node. This is the default anyway.

#### register(o:[Delay](#delay-948)) -> [Integer](#integer-769)

<a name="register-419"></a>

Register a new variate. This is a callback function typically called
within the Delay class itself.

*Returns* an assigned to the variate.

#### trigger()

<a name="trigger-427"></a>

Trigger an event.


## DelayReal

<a name="delayreal-467"></a>

  * Inherits from *[Delay](#delay-948)*

Abstract delay variate with real value.

| Assignment | Description |
| --- | --- |
| *[Real](#real-692)* | Value assignment. |
| *[String](#string-790)* | String assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-692)* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-692)* | Value. |
| *w:[Real](#real-692)* | Weight. |


## DelayRealVector

<a name="delayrealvector-510"></a>

  * Inherits from *[Delay](#delay-948)*

Abstract delay variate with real vector value.

`D` Number of dimensions.

| Assignment | Description |
| --- | --- |
| *[Real](#real-692)\[\_\]* | Value assignment. |

| Conversion | Description |
| --- | --- |
| *[Real](#real-692)\[\_\]* | Value conversion. |

| Member Variable | Description |
| --- | --- |
| *x:[Real](#real-692)\[\_\]* | Value. |
| *w:[Real](#real-692)* | Weight. |


## Gamma

<a name="gamma-317"></a>

Gamma distribution.

| Member Variable | Description |
| --- | --- |
| *k:[Real](#real-692)* | Shape. |
| *θ:[Real](#real-692)* | Scale. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-314) | Simulate. |
| [observe](#observe-316) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-692)) -> [Real](#real-692)

<a name="observe-316"></a>

Observe.

#### simulate() -> [Real](#real-692)

<a name="simulate-314"></a>

Simulate.


## Gaussian

<a name="gaussian-307"></a>

  * Inherits from *[DelayReal](#delayreal-467)*

Gaussian distribution.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-692)* | Mean. |
| *σ2:[Real](#real-692)* | Variance. |


## MultivariateGaussian

<a name="multivariategaussian-444"></a>

  * Inherits from *[DelayRealVector](#delayrealvector-510)*

Multivariate Gaussian distribution.

`D` Number of dimensions.

| Member Variable | Description |
| --- | --- |
| *μ:[Real](#real-692)\[\_\]* | Mean. |
| *Σ:[Real](#real-692)\[\_,\_\]* | Covariance matrix. |


## Uniform

<a name="uniform-785"></a>

Uniform distribution.

| Member Variable | Description |
| --- | --- |
| *l:[Real](#real-692)* | Lower bound. |
| *u:[Real](#real-692)* | Upper bound. |

| Member Function | Brief description |
| --- | --- |
| [simulate](#simulate-782) | Simulate. |
| [observe](#observe-784) | Observe. |


### Member Function Details

#### observe(x:[Real](#real-692)) -> [Real](#real-692)

<a name="observe-784"></a>

Observe.

#### simulate() -> [Real](#real-692)

<a name="simulate-782"></a>

Simulate.

