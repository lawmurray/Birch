/**
 * @file
 */
#include "bi/io/cpp/CppDispatcherGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppDispatcherGenerator::CppDispatcherGenerator(std::ostream& base,
    const int level, const bool header) :
    CppParameterGenerator(base, level, header) {
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
  int i;
  if (!header) {
    start("struct ");
    middle("visitor_" << o->name << "_" << o->number << '_');
    finish(" : public boost::static_visitor<" << o->type << "> {");
    in();

    if (o->types.size() >= 1) {
      start("template<");
    }
    i = 0;
    for (auto iter = o->types.begin(); iter != o->types.end(); ++iter) {
      if (i > 0) {
        middle(", ");
      }
      middle("class T" << ++i);
    }
    if (o->types.size() >= 1) {
      finish(">");
    }
    start(o->type << " operator()(");
    i = 0;
    for (auto iter = o->types.begin(); iter != o->types.end(); ++iter) {
      if (i > 0) {
        middle(", ");
      }
      ++i;
      middle('T' << i << "&& o" << i);
    }
    finish(") const {");
    in();
    genBody(o);
    out();
    line("}");
    out();
    line("};");
  }

  if (header) {
    middle("static ");
  }
  start(o->type << ' ');
  start("dispatch_" << o->name << "_" << o->number << "_(");
  i = 0;
  for (auto iter = o->types.begin(); iter != o->types.end(); ++iter) {
    if (i > 0) {
      middle(", ");
    }
    ++i;
    if (!(*iter)->assignable) {
      middle("const ");
    }
    middle(*iter << "& o" << i);
  }
  middle(')');
  if (header) {
    finish(";");
  } else {
    finish(" {");
    in();
    start("return boost::apply_visitor(");
    middle("visitor_" << o->name << "_" << o->number << "_()");
    i = 0;
    for (auto iter = o->types.begin(); iter != o->types.end(); ++iter) {
      middle(", o" << ++i);
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
  for (auto iter = o->funcs.rbegin(); iter != o->funcs.rend(); ++iter) {
    const FuncParameter* func = *iter;
    start("try { ");
    if (!func->result->type->isEmpty()) {
      middle("return ");
    }
    if (func->isBinary() && isTranslatable(func->name->str())) {
      genArg(func->getLeft(), 1);
      middle(' ' << o->name << ' ');
      genArg(func->getRight(), 2);
    } else if (func->isUnary() && isTranslatable(func->name->str())) {
      middle(o->name << ' ' << func->getRight());
      genArg(func->getLeft(), 1);
    } else {
      middle("bi::" << func->name << '(');
      auto iter1 = func->parens->begin();
      int i = 0;
      while (iter1 != func->parens->end()) {
        if (i > 0) {
          middle(", ");
        }
        genArg(*iter1, ++i);
        ++iter1;
      }
      middle(')');
    }
    middle(';');
    if (func->result->type->isEmpty()) {
      middle(" return boost::blank();");
    }
    finish(" } catch (std::bad_cast e) {}");
  }
  line("throw std::bad_cast();");
}

void bi::CppDispatcherGenerator::genArg(const Expression* o, const int i) {
  middle("bi::cast<" << o->type << ">(o" << i << ')');
}
