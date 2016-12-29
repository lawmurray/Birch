/**
 * @file
 */
#include "bi/io/cpp/CppTemplateParameterGenerator.hpp"

bi::CppTemplateParameterGenerator::CppTemplateParameterGenerator(
    std::ostream& base, const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    n(1) {
  //
}

void bi::CppTemplateParameterGenerator::visit(const ModelReference* o) {
  //middle("class Group" << n);
  //if (header) {
  //  middle(" = StackGroup");
  //}
  //if (o->count() > 0) {
  //  middle(", class Frame" << n);
  //}
}

void bi::CppTemplateParameterGenerator::visit(const FuncParameter* o) {
  //if (o->inputs.size() > 0) {
  //  start("template<");
  //  for (auto iter = o->inputs.begin(); iter != o->inputs.end();
  //      ++iter, ++n) {
  //    const VarParameter* param = *iter;
  //    if (iter != o->inputs.begin()) {
  //      middle(", ");
  //    }
  //    middle(param->type);
  //  }
  //  finish('>');
  //}
}
