/**
 * @file
 */
#include "bi/io/cpp/CppDispatchGenerator.hpp"

bi::CppDispatchGenerator::CppDispatchGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppDispatchGenerator::visit(const FuncParameter* o) {
  for (auto iter1 = o->inputs.begin(); iter1 != o->inputs.end(); ++iter1) {
    const VarParameter* param = *iter1;
    const RandomType* type = dynamic_cast<const RandomType*>(param->type->strip());
    if (type) {
      line("if (!" << param->name << ".isMissing()) {");
      in();
      start("return " << o->mangled << '(');
      for (auto iter2 = o->inputs.begin(); iter2 != o->inputs.end(); ++iter2) {
        if (iter2 != o->inputs.begin()) {
          middle(", ");
        }
        if (iter2 == iter1) {
          /* cast to variate type */
          middle('(' << type->left << "&)");
        }
        middle((*iter2)->name);
      }
      finish(");");
      out();
      line("}");
    }
  }
}
