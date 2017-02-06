/**
 * @file
 */
#include "bi/io/cpp/CppRandomGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppRandomGenerator::CppRandomGenerator(std::ostream& base,
    const int level, const bool header) :
    indentable_ostream(base, level, header) {
  //
}

void bi::CppRandomGenerator::visit(const ModelParameter* o) {
  /* variate copy assignment operator */
  if (!header) {
    line("template<class Group>");
    start("bi::model::");
  } else {
    start("");
  }
  middle(o->name->str() << "<Group>& ");
  if (!header) {
    middle("bi::model::" << o->name->str() << "<Group>::");
  }
  middle("operator=(const variate_type& o_)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    line("x_ = o_;");
    line("missing_ = false;");
    line("return *this;");
    out();
    line("}\n");
  }

  /* variate move assignment operator */
  if (!header) {
    line("template<class Group>");
    start("bi::model::");
  } else {
    start("");
  }
  middle(o->name->str() << "<Group>& ");
  if (!header) {
    middle("bi::model::" << o->name->str() << "<Group>::");
  }
  middle("operator=(variate_type&& o_)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    line("x_ = o_;");
    line("missing_ = false;");
    line("return *this;");
    out();
    line("}\n");
  }

  /* variate cast operator */
  if (!header) {
    line("template<class Group>");
    start("bi::model::" << o->name->str() << "<Group>::");
  } else {
    start("");
  }
  middle("operator variate_type&()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    line("if (missing_) {");
    in();
    line("randomStack.pop(pos_);");
    out();
    line("}");
    line("assert(!missing_);");
    line("return x_;");
    out();
    line("}\n");
  }

  /* init function */
  if (!header) {
    line("template<class Group>");
  }
  start("void ");
  if (!header) {
    middle("bi::model::" << o->name->str() << "<Group>::");
  }
  middle(
      "init(const " << o->name->str() << "<Group>& m_, const lambda_type& pull_, const lambda_type& push_)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    line("operator=(m_);");
    line("if (!missing_) {");
    in();
    line("/* push immediately */");
    line("push_();");
    out();
    line("} else {");
    in();
    line("/* lazy sampling */");
    line("pos_ = randomStack.push(pull_, push_);");
    out();
    line("}");
    out();
    line("}\n");
  }
}
