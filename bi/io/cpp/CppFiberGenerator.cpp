/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

bi::CppFiberGenerator::CppFiberGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    point(0) {
  //
}

void bi::CppFiberGenerator::visit(const Fiber* o) {
  fiberType = dynamic_cast<const FiberType*>(o->returnType);
  assert(fiberType);

  /* generate a unique name (within this scope) for the state of the fiber */
  std::stringstream base;
  bih_ostream buf(base);
  buf << o->params;
  std::string baseName = internalise(o->name->str()) + '_' + encode32(base.str());
  std::string stateName = baseName + "_FiberState_";

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class for state */
  if (header) {
    start("class " << stateName << " final : ");
    finish("public libbirch::FiberState<" << fiberType->yieldType << "> {");
    line("public:");
    in();
    line("using class_type_ = " << stateName << ';');
    line("using super_type_ = libbirch::FiberState<" << fiberType->yieldType << ">;\n");
    for (auto o : params) {
      line(o->type << ' ' << o->name << ';');
    }
    for (auto o : locals) {
      if (o->has(IN_FIBER)) {
        start(o->type << ' ');
        finish(getName(o->name->str(), o->number) << ';');
      }
    }
  }

  /* constructor */
  if (!header) {
    genSourceLine(o->loc);
    start("bi::" << stateName << "::");
  } else {
    start("");
  }
  middle(stateName << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" :");
    in();
    in();
    genSourceLine(o->loc);
    start("super_type_(" << (yields.size() + 1) << ')');
    for (auto param : params) {
      finish(',');
      genSourceLine(param->loc);
      start(param->name << '(' << param->name << ')');
    }
    finish(" {");
    out();
    line("//");
    out();
    line("}\n");
  }

  /* deep copy constructor */
  if (!header) {
    genSourceLine(o->loc);
    start("bi::" << stateName << "::");
  } else {
    start("");
  }
  middle(stateName << "(libbirch::Label* label, const " << stateName << "& o)");
  if (header) {
    finish(";\n");
  } else {
    finish(" :");
    in();
    in();
    genSourceLine(o->loc);
    start("super_type_(label, o)");
    for (auto o : params) {
      finish(',');
      genSourceLine(o->loc);
      if (o->type->isValue()) {
        start(o->name << "(o." << o->name << ')');
      } else {
        start(o->name << "(label, o." << o->name << ')');
      }
    }
    for (auto o : locals) {
      if (o->has(IN_FIBER)) {
        auto name = getName(o->name->str(), o->number);
        finish(',');
        genSourceLine(o->loc);
        if (o->type->isValue()) {
          start(name << "(o." << name << ')');
        } else {
          start(name << "(label, o." << name << ')');
        }
      }
    }
    out();
    out();
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }

  /* copy constructor, destructor, assignment operator */
  if (header) {
    genSourceLine(o->loc);
    line("virtual ~" << stateName << "() = default;");
    genSourceLine(o->loc);
    line(stateName << "(const " << stateName << "&) = delete;");
    genSourceLine(o->loc);
    line(stateName << "& operator=(const " << stateName << "&) = delete;");
  }

  /* clone function */
  if (header) {
    genSourceLine(o->loc);
    line("virtual " << stateName << "* clone_() const {");
    in();
    genSourceLine(o->loc);
    line("return libbirch::clone_object<" << stateName << ">(this);");
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* name function */
  if (header) {
    genSourceLine(o->loc);
    line("virtual const char* getClassName() const {");
    in();
    genSourceLine(o->loc);
    line("return \"" << stateName << "\";");
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* freeze function */
  genSourceLine(o->loc);
  if (header) {
    start("virtual void ");
  } else {
    start("void bi::" << stateName << "::");
  }
  middle("doFreeze_()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genSourceLine(o->loc);
    line("super_type_::doFreeze_();");
    if (!fiberType->yieldType->isValue()) {
      genSourceLine(o->loc);
      line("value_.freeze();");
    }
    for (auto o : params) {
      if (!o->type->isValue() && o->has(IN_FIBER)) {
        genSourceLine(o->loc);
        line(o->name << ".freeze();");
      }
    }
    for (auto o : locals) {
      if (!o->type->isValue() && o->has(IN_FIBER)) {
        genSourceLine(o->loc);
        line(getName(o->name->str(), o->number) << ".freeze();");
      }
    }
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* thaw function */
  genSourceLine(o->loc);
  if (header) {
    start("virtual void ");
  } else {
    start("void bi::" << stateName << "::");
  }
  middle("doThaw_(libbirch::Label* label_)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genSourceLine(o->loc);
    line("super_type_::doThaw_(label_);");
    if (!fiberType->yieldType->isValue()) {
      genSourceLine(o->loc);
      line("value_.thaw(label_);");
    }
    for (auto o : params) {
      if (!o->type->isValue() && o->has(IN_FIBER)) {
        genSourceLine(o->loc);
        line(o->name << ".thaw(label_);");
      }
    }
    for (auto o : locals) {
      if (!o->type->isValue() && o->has(IN_FIBER)) {
        genSourceLine(o->loc);
        line(getName(o->name->str(), o->number) << ".thaw(label_);");
      }
    }
    genSourceLine(o->loc);
    out();
    line("}\n");
  }

  /* finish function */
  genSourceLine(o->loc);
  if (header) {
    start("virtual void ");
  } else {
    start("void bi::" << stateName << "::");
  }
  middle("doFinish_()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genSourceLine(o->loc);
    line("super_type_::doFinish_();");
    if (!fiberType->yieldType->isValue()) {
      genSourceLine(o->loc);
      line("value_.finish();");
    }
    for (auto o : params) {
      if (!o->type->isValue() && o->has(IN_FIBER)) {
        genSourceLine(o->loc);
        line(o->name << ".finish();");
      }
    }
    for (auto o : locals) {
      if (!o->type->isValue() && o->has(IN_FIBER)) {
        genSourceLine(o->loc);
        line(getName(o->name->str(), o->number) << ".finish();");
      }
    }
    genSourceLine(o->loc);
    out();
    line("}");
  }

  /* query function */
  genSourceLine(o->loc);
  if (header) {
    start("virtual ");
  } else {
    start("");
  }
  middle("bool ");
  if (!header) {
    middle("bi::" << stateName << "::");
  }
  middle("query()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    for (auto iter = o->typeParams->begin(); iter != o->typeParams->end();
        ++iter) {
      auto param = dynamic_cast<const Generic*>(*iter);
      assert(param);
      genSourceLine(o->loc);
      line("using " << param->name << " [[maybe_unused]] = " <<
          param->type << ';');
    }
    genSourceLine(o->loc);
    line("libbirch_declare_local_");
    genSourceLine(o->loc);
    line("switch (local->point_) {");
    in();
    for (int s = 0; s <= yields.size(); ++s) {
      genSourceLine(o->loc);
      line("case " << s << ": goto POINT" << s << "_;");
    }
    genSourceLine(o->loc);
    line("default: goto END_;");
    genSourceLine(o->loc);
    out();
    line('}');
    genSourceLine(o->loc);
    line("POINT0_:");
    ++point;

    *this << o->braces->strip();

    line("END_:");
    genSourceLine(o->loc);
    line("local->point_ = " << (yields.size() + 1) << ';');
    genSourceLine(o->loc);
    line("return false;");
    genSourceLine(o->loc);
    out();
    finish("}\n");
  }
  if (header) {
    out();
    line("};\n");
  }

  /* initialisation function */
  genSourceLine(o->loc);
  start(fiberType << ' ');
  if (!header) {
    middle("bi::");
  }
  middle(o->name << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genSourceLine(o->loc);
    start("return libbirch::make_fiber<" << stateName << ">(");
    for (auto iter = params.begin(); iter != params.end(); ++iter) {
      if (iter != params.begin()) {
        middle(", ");
      }
      middle((*iter)->name);
    }
    finish(");");
    genSourceLine(o->loc);
    out();
    line("}\n");
  }
}

void bi::CppFiberGenerator::visit(const Return* o) {
  genSourceLine(o->loc);
  line("goto END_;");
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  genTraceLine(o->loc);
  start("local->value_ = " << o->single << ';');
  genSourceLine(o->loc);
  line("local->point_ = " << point << ';');
  genSourceLine(o->loc);
  line("return true;");
  genSourceLine(o->loc);
  line("POINT" << point << "_: ;");
  ++point;
}

void bi::CppFiberGenerator::visit(const For* o) {
  genTraceLine(o->loc);
  start("for (");
  auto forVar = dynamic_cast<const LocalVariable*>(o->index);
  if (forVar && !forVar->has(IN_FIBER)) {
    middle("auto ");
  }
  middle(o->index << " = " << o->from << "; ");
  middle(o->index << " <= " << o->to << "; ");
  finish("++" << o->index << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void bi::CppFiberGenerator::visit(const Parameter* o) {
  if (!o->has(IN_FIBER)) {
    CppBaseGenerator::visit(o);
  } else {
    middle("const " << o->type);
    if (o->type->isArray() || o->type->isClass()) {
      middle('&');
    }
    middle(' ' << o->name);
    if (!o->value->isEmpty()) {
      middle(" = " << o->value);
    }
  }
}

void bi::CppFiberGenerator::visit(const LocalVariable* o) {
  if (!o->has(IN_FIBER)) {
    CppBaseGenerator::visit(o);
  } else {
    genTraceLine(o->loc);
    auto name = getName(o->name->str(), o->number);
    if (!o->value->isEmpty()) {
      middle("local->" << name << " = " << o->value);
    } else if (!o->args->isEmpty()) {
      middle("local->" << name << " = ");
      middle(o->type << "(" << o->args << ")");
    } else if (!o->brackets->isEmpty()) {
      middle("local->" << name << " = ");
      middle(o->type << "(libbirch::make_shape(" << o->brackets << "))");
    } else {
      middle("local->" << name << " = " << o->type << "()");
    }
    finish(';');
  }
}

void bi::CppFiberGenerator::visit(const NamedExpression* o) {
  middle("local->" << o->name);
}

void bi::CppFiberGenerator::visit(const LambdaFunction* o) {
  CppBaseGenerator aux(base, level, false);
  aux << o;
}

std::string bi::CppFiberGenerator::getName(const std::string& name,
    const int number) {
  std::stringstream buf;
  std::string result;
  auto iter = names.find(number);
  if (iter == names.end()) {
    auto count = counts.find(name);
    if (count == counts.end()) {
      buf << internalise(name);
      result = buf.str();
      counts.insert(std::make_pair(name, 1));
    } else {
      buf << internalise(name) << count->second << '_';
      result = buf.str();
      ++count->second;
    }
    names.insert(std::make_pair(number, result));
  } else {
    result = iter->second;
  }
  return result;
}
