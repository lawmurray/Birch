/**
 * Run the package.
 */
program run() {
  code:Integer <- 0;

  /* simulate a new data set */
  code <- code + system("birch sample --model Multi --config config/simulate.json --input input/simulate.json --output output/simulate.json");
  code <- code + system("birch draw --input output/simulate.json --output figs/simulate.pdf");
  code <- code + system("birch data --input output/simulate.json --output input/filter.json");

  /* run the particle filter on the data set */
  code <- code + system("birch sample --model Multi --config config/filter.json --input input/filter.json --output output/filter.json");
  code <- code + system("birch draw --input output/filter.json --output figs/filter.pdf");
  
  exit(code);
}
