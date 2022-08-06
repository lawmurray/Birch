/**
 * @file
 * 
 * Base source for standalone programs.
 */
#include <iostream>
#include <string>

typedef int prog_t(int argc, char** argv);
extern "C" prog_t* retrieve_program(const std::string&);

int main(int argc, char** argv) {
  if (argc > 1) {
    std::string prog = argv[1];
    prog_t* f = retrieve_program(prog);
    if (f) {
      return f(argc - 1, argv + 1);
    } else {
      std::cerr << "no program " << prog << std::endl;
    }
  } else {
    std::cerr << "no program specified" << std::endl;
  }
  return EXIT_FAILURE;
}
