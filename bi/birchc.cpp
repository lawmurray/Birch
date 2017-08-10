/**
 * @file
 *
 * The compiler program.
 */
#include "bi/build/Compiler.hpp"

#include <iostream>
#include <sstream>

bi::Compiler* compiler = nullptr;  // global variable needed by GNU Bison parser
std::stringstream raw;  // stream for raw code and /** ... */ comments

int main(int argc, char** argv) {
  using namespace bi;

  try {
    compiler = new Compiler(argc, argv);
    compiler->parse();
    compiler->resolve();
    compiler->gen();
  } catch (Exception& e) {
    std::cerr << e.msg << std::endl;
    return 1;
  }
  delete compiler;

  return 0;
}
