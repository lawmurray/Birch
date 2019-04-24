/**
 * Run the package.
 */
program run() {
  exit(system("birch sample --config config/yap_dengue.json --input input/yap_dengue.json --output output/yap_dengue.json"));
}
