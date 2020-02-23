/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/io/cpp/CppResumeGenerator.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

bi::CppFiberGenerator::CppFiberGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppFiberGenerator::visit(const Fiber* o) {
}
