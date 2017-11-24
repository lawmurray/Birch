/**
 * @file
 */
#include "libubjpp/json/JSONDriver.hpp"
#include "libubjpp/json/JSONGenerator.hpp"
#include "libubjpp/value.hpp"

#include <iostream>

int main(int argc, char** argv) {
  using namespace libubjpp;

  for (int i = 1; i < argc; ++i) {
    JSONDriver driver;
    JSONGenerator generator(std::cout);
    generator.write(driver.parse(argv[i]));
  }
  return 0;
}
