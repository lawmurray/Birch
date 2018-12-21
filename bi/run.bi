/**
 * Run the package.
 */
program run() {
  /* simulate a new data set */
  system("birch sample --model Multi --input input/simulate.json --output output/simulate.json");
  system("birch draw --input output/simulate.json --output figs/simulate.pdf");
  system("birch data --input output/simulate.json --output input/filter.json");

  /* run the particle filter on the data set */
  system("birch sample --model Multi --input input/filter.json --output output/filter.json --config config/filter.json --diagnostic output/diagnostic.json");
  system("birch draw --input output/filter.json --output figs/filter.pdf");
}
