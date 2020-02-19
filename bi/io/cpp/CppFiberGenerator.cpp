/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/io/cpp/CppResumeGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppFiberGenerator::CppFiberGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppFiberGenerator::visit(const Fiber* o) {
  /* initialization function */
  genTemplateParams(o);
  genSourceLine(o->loc);
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::");
  }
  middle(o->name << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceLine(o->loc);
    start("return make_fiber_" << o->name << "_0_(");
    middle("libbirch::make_tuple(");
    for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
      if (iter != o->params->begin()) {
        middle(", ");
      }
      auto param = dynamic_cast<const Parameter*>(*iter);
      assert(param);
      middle(param->name);
    }
    finish("));");
    out();
    line("}\n");
  }

  /* start function */
  CppResumeGenerator auxResume(o, base, level, header);
  auxResume << o->start;

  /* resume functions */
  Gatherer<Yield> yields;
  o->accept(&yields);
  for (auto yield : yields) {
    if (yield->resume) {
      CppResumeGenerator auxResume(o, base, level, header);
      auxResume << yield->resume;
    }
  }
}
