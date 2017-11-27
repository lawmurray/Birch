/**
 * @file
 */
#include "bi/io/cpp/CppRawGenerator.hpp"

#include "bi/io/cpp/CppBaseGenerator.hpp"

bi::CppRawGenerator::CppRawGenerator(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level),
    header(header) {
  //
}

void bi::CppRawGenerator::visit(const Function* o) {
  // don't go any further
}

void bi::CppRawGenerator::visit(const Fiber* o) {
  // don't go any further
}

void bi::CppRawGenerator::visit(const Program* o) {
  // don't go any further
}

void bi::CppRawGenerator::visit(const Class* o) {
  // don't go any further
}

void bi::CppRawGenerator::visit(const Raw* o) {
  CppBaseGenerator aux(base, level, header);
  aux << o;
}
