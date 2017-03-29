/**
 * @file
 */
#include "bi/io/cpp/CppDispatcherGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/io/cpp/CppTemplateParameterGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppDispatcherGenerator::CppDispatcherGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppDispatcherGenerator::visit(const File* o) {
  for (auto iter = o->scope->dispatchers.begin();
      iter != o->scope->dispatchers.end(); ++iter) {
    *this << iter->second;
  }
}

void bi::CppDispatcherGenerator::visit(const Dispatcher* o) {
  /* a dispatcher uses the boost::variant mechanism to determine the precise
   * types of all variant arguments, then bi::cast handles the rest */
  bool before;
  int i;

  /* visitor struct for determining the precise types of variant arguments */
  if (!header) {
    start("struct ");
    middle("visitor_" << o->name << "_" << o->number << '_');
    finish(" : public boost::static_visitor<" << o->type << "> {");
    in();
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      const Expression* param = *iter;
      if (!param->type->isVariant()) {
        start("");
        if (!param->type->assignable) {
          middle("const ");
        }
        finish(param << ';');
      }
    }
    line("");

    start("visitor_" << o->name << "_" << o->number << "_(");
    before = false;
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      const Expression* param = *iter;
      if (!param->type->isVariant()) {
        if (before) {
          middle(", ");
        }
        if (!param->type->assignable) {
          middle("const ");
        }
        middle(param);
        before = true;
      }
    }
    finish(") :");
    in();
    in();
    before = false;
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      const VarParameter* param = dynamic_cast<const VarParameter*>(*iter);
      if (!param->type->isVariant()) {
        if (before) {
          finish(',');
        }
        start(param->name << '(' << param->name << ')');
        before = true;
      }
    }
    finish(" {");
    out();
    line("//");
    out();
    line("}\n");

    start("template<");
    i = 1;
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      const Expression* param = *iter;
      if (param->type->isVariant()) {
        if (i > 1) {
          middle(", ");
        }
        middle("class T" << i++);
      }
    }
    finish(">");
    start(o->type << " operator()(");
    i = 1;
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      const VarParameter* param = dynamic_cast<const VarParameter*>(*iter);
      if (param->type->isVariant()) {
        if (i > 1) {
          middle(", ");
        }
        middle('T' << i << "&& " << param->name);
      }
    }
    finish(") const {");
    in();
    genBody(o);
    out();
    line("}");
    out();
    line("};");
  }

  /* template parameters */
  CppTemplateParameterGenerator auxTemplateParameter(base, level, header);
  auxTemplateParameter << o;

  if (header) {
    middle("static ");
  }

  /* return type */
  start(o->type << ' ');

  /* name */
  start("dispatch_" << o->name << "_" << o->number << '_');

  /* parameters */
  CppParameterGenerator auxParameter(base, level, header);
  auxParameter << o;

  /* body */
  if (header) {
    finish(";");
  } else {
    finish(" {");
    in();
    start("return boost::apply_visitor(");
    middle("visitor_" << o->name << "_" << o->number << "_(");
    bool before = false;
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      const VarParameter* param = dynamic_cast<const VarParameter*>(*iter);
      if (!param->type->isVariant()) {
        if (before) {
          middle(", ");
        }
        middle(param->name);
        before = true;
      }
    }
    middle(")");
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      const VarParameter* param = dynamic_cast<const VarParameter*>(*iter);
      if (param->type->isVariant()) {
        middle(", " << param->name);
      }
    }
    finish(");");
    out();
    line("}\n");
  }
}

void bi::CppDispatcherGenerator::visit(const VarParameter* o) {
  middle(o->name);
}

void bi::CppDispatcherGenerator::genBody(const Dispatcher* o) {
  /* try functions, in topological order from most specific */
  for (auto iter = o->funcs.begin(); iter != o->funcs.end(); ++iter) {
    const FuncParameter* func = *iter;
    auto iter1 = o->parens->begin();
    int i = 1;

    start("try { return ");
    if (func->isBinary() && isTranslatable(func->name->str())) {
      genArg(*(iter1++), i++);
      middle(' ' << o->name << ' ');
      genArg(*(iter1++), i++);
    } else if (func->isUnary() && isTranslatable(func->name->str())) {
      middle(o->name << ' ');
      genArg(*(iter1++), i++);
    } else {
      middle("bi::" << func->name << '(');
      while (iter1 != o->parens->end()) {
        if (iter1 != o->parens->begin()) {
          middle(", ");
        }
        genArg(*(iter1++), i++);
      }
      middle(")");
    }
    finish("; } catch (std::bad_cast e) {}");
  }
  line("throw std::bad_cast();");
}

void bi::CppDispatcherGenerator::genArg(const Expression* o, const int i) {
  middle("bi::cast<");
  if (!o->type->assignable) {
    middle("const ");
  }
  middle(o->type);
  middle("&>(o" << i << ')');
}
