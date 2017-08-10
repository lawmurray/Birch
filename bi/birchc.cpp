/**
 * @file
 *
 * The compiler program.
 */
#include "bi/build/Compiler.hpp"

#include <iostream>

int main(int argc, char** argv) {
  try {
    bi::Compiler compiler(argc, argv);
    compiler.parse();
    compiler.resolve();
    compiler.gen();
  } catch (bi::Exception& e) {
    std::cerr << e.msg << std::endl;
    return 1;
  }
  return 0;
}
