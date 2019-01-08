/**
 * Run the package.
 */
program run() {
  code:Integer <- 0;
  code <- code + system("birch sample --model LinearRegressionModel --input input/bike_share.json --output output/linear_regression.json");
  code <- code + system("birch sample --model LinearGaussianModel --input input/linear_gaussian.json --output output/linear_gaussian.json");
  code <- code + system("birch sample --model MixedGaussianModel --config config/mixed_gaussian.json --input input/mixed_gaussian.json --output output/mixed_gaussian.json");
  code <- code + system("birch sample --model SIRModel --config config/sir_model.json --input input/russian_influenza.json --output output/sir_model.json");
  exit(code);
}
