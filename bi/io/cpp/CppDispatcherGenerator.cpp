/**
 * @file
 */
#include "bi/io/cpp/CppDispatcherGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/io/cpp/CppTemplateParameterGenerator.hpp"
#include "bi/io/cpp/misc.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/capture/ArgumentCapturer.hpp"

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

  Gatherer<VarParameter> gatherer;
  o->parens->accept(&gatherer);
  bool before;
  int i;

  /* visitor struct for determining the precise types of variant arguments */
  if (!header && o->hasVariant()) {
    start("struct ");
    middle("visitor_" << o->mangled << "_" << o->number << '_');
    finish(" : public boost::static_visitor<" << o->type << "> {");
    in();
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      VarParameter* param = *iter;
      if (!param->type->isVariant()) {
        if (!param->type->assignable) {
          middle("const ");
        }
        line(param->type << "& " << param->name << ';');
      }
    }
    line("");

    start("visitor_" << o->mangled << "_" << o->number << "_(");
    before = false;
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      VarParameter* param = *iter;
      if (!param->type->isVariant()) {
        if (before) {
          middle(", ");
        }
        if (!param->type->assignable) {
          middle("const ");
        }
        middle(param->type << "& " << param->name);
        before = true;
      }
    }
    finish(") :");
    in();
    in();
    before = false;
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      VarParameter* param = *iter;
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
    Gatherer<VarParameter> gatherer;
    o->parens->accept(&gatherer);
    i = 1;
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      VarParameter* param = *iter;
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
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      VarParameter* param = *iter;
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
  start("dispatch_" << o->mangled << "_" << o->number << '_');

  /* parameters */
  CppParameterGenerator auxParameter(base, level, header);
  auxParameter << o;

  /* body */
  if (header) {
    finish(";");
  } else {
    finish(" {");
    in();
    if (o->hasVariant()) {
      start("return boost::apply_visitor(");
      middle("visitor_" << o->mangled << "_" << o->number << "_(");
      bool before = false;
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        VarParameter* param = *iter;
        if (!param->type->isVariant()) {
          if (before) {
            middle(", ");
          }
          middle(param->name);
          before = true;
        }
      }
      middle(")");
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        VarParameter* param = *iter;
        if (param->type->isVariant()) {
          middle(", " << param->name);
        }
      }
      finish(");");
    } else {
      genBody(o);
    }
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
    FuncParameter* func = *iter;
    ArgumentCapturer capturer(o->parens.get(), func->parens.get());
    auto iter1 = capturer.begin();

    start("try { return ");
    if (func->isBinary() && isTranslatable(func->name->str())
        && !func->parens->isRich()) {
      genArg(iter1->first, iter1->second);
      ++iter1;
      middle(' ' << translate(o->name->str()) << ' ');
      genArg(iter1->first, iter1->second);
      ++iter1;
    } else if (func->isUnary() && isTranslatable(func->name->str())
        && !func->parens->isRich()) {
      middle(translate(o->name->str()) << ' ');
      genArg(iter1->first, iter1->second);
      ++iter;
    } else {
      middle("bi::" << func->mangled << '(');
      for (; iter1 != capturer.end(); ++iter1) {
        if (iter1 != capturer.begin()) {
          middle(", ");
        }
        genArg(iter1->first, iter1->second);
      }
      middle(")");
    }
    finish("; } catch (std::bad_cast e) {}");
  }

  /* defer to parent dispatcher */
  if (o->parent) {
    bool result = o->parens->possibly(*o->parent->parens);
    assert(result);

    Gatherer<VarParameter> gatherer;
    o->parent->parens->accept(&gatherer);

    start(
        "return dispatch_" << o->parent->mangled << "_" << o->parent->number << "_(");
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      if (iter != gatherer.begin()) {
        middle(", ");
      }
      middle((*iter)->arg);
    }
    finish(");");
  } else {
    line("throw std::bad_cast();");
  }
}

void bi::CppDispatcherGenerator::genArg(const Expression* arg,
    const VarParameter* param) {
  Expression* arg1 = const_cast<Expression*>(arg);
  VarParameter* param1 = const_cast<VarParameter*>(param);

  if (!arg1->type->equals(*param1->type)) {
    middle("bi::cast<");
    if (!param1->type->assignable) {
      middle("const ");
    }
    if (!arg1->type->isLambda() && param1->type->isLambda()) {
      const LambdaType* lambda =
          dynamic_cast<const LambdaType*>(param1->type.get());
      assert(lambda);
      middle(lambda->result);
    } else {
      middle(param1->type);
    }
    middle("&>(" << arg1 << ')');
  } else {
    middle(arg1);
  }
}
