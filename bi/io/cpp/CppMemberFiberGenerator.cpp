/**
 * @file
 */
#include "bi/io/cpp/CppMemberFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

bi::CppMemberFiberGenerator::CppMemberFiberGenerator(const Class* type,
    std::ostream& base, const int level, const bool header) :
    CppFiberGenerator(base, level, header),
    type(type) {
  //
}

void bi::CppMemberFiberGenerator::visit(const MemberFiber* o) {
  yieldType = o->returnType->unwrap();

  /* generate a unique name (within this scope) for the state of the fiber */
  std::stringstream base;
  bih_ostream buf(base);
  buf << o->params;
  std::string baseName = internalise(o->name->str()) + encode32(base.str());
  std::string stateName = baseName + "_FiberState_";

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class for state */
  if (header) {
    start("class " << stateName << " final : ");
    finish("public libbirch::FiberState<" << yieldType << "> {");
    line("public:");
    in();
    line("using class_type_ = " << stateName << ';');
    line("using super_type_ = libbirch::FiberState<" << yieldType << ">;\n");
    start("libbirch::LazySharedPtr<bi::type::" << type->name);
    genTemplateArgs(type);
    finish("> self;");
    for (auto param : params) {
      line(param->type << ' ' << param->name << ';');
    }
    for (auto local : locals) {
      start(local->type << ' ');
      finish(getName(local->name->str(), local->number) << ';');
    }
  }

  /* constructor */
  start("");
  if (!header) {
    genTraceLine(o->loc);
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle(stateName << "(libbirch::Label* context_, const libbirch::LazySharedPtr<");
  middle(type->name);
  genTemplateArgs(type);
  middle(">& self");
  if (!o->params->isEmpty()) {
    middle(", " << o->params);
  }
  middle(')');
  if (header) {
    finish(';');
  } else {
    finish(" :");
    in();
    in();
    genTraceLine(o->loc);
    start("super_type_(context_, " << (yields.size() + 1) << ')');
    finish(',');
    genTraceLine(o->loc);
    start("self(context_, self)");
    for (auto param : params) {
      finish(',');
      genTraceLine(param->loc);
      start(param->name << '(');
      if (!param->type->isValue()) {
        middle("context_, ");
      }
      middle(param->name << ')');
    }
    finish(" {");
    out();
    line("//");
    out();
    line("}\n");
  }

  /* deep copy constructor */
  if (!header) {
    genTraceLine(o->loc);
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  } else {
    start("");
  }
  middle(stateName << "(libbirch::Label* context, libbirch::Label* label, const " << stateName << "& o)");
  if (header) {
    finish(";\n");
  } else {
    finish(" :");
    in();
    in();
    genTraceLine(o->loc);
    start("super_type_(context, label, o)");
    finish(',');
    genTraceLine(o->loc);
    start("self(context, label, o.self)");
    for (auto o : params) {
      if (!o->type->isValue()) {
        finish(',');
        genTraceLine(o->loc);
        if (o->type->isValue()) {
          start(o->name << "(o." << o->name << ')');
        } else {
          start(o->name << "(context, label, o." << o->name << ')');
        }
      }
    }
    for (auto o : locals) {
      auto name = getName(o->name->str(), o->number);
      finish(',');
      genTraceLine(o->loc);
      if (o->type->isValue()) {
        start(name << "(o." << name << ')');
      } else {
        start(name << "(context, label, o." << name << ')');
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
    line("virtual ~" << stateName << "() = default;");
    line(stateName << "(const " << stateName << "&) = delete;");
    line(stateName << "& operator=(const " << stateName << "&) = delete;");
  }

  /* clone function */
  if (header) {
    line("virtual " << stateName << "* clone_(libbirch::Label* context_) const {");
    in();
    line("return libbirch::clone_object<" << stateName << ">(context_, this);");
    out();
    line("}\n");
  }

  /* name function */
  if (header) {
    line("virtual const char* name_() const {");
    in();
    line("return \"" << stateName << "\";");
    out();
    line("}\n");
  }

  /* freeze function */
  if (header) {
    start("virtual void ");
  } else {
    genTraceLine(o->loc);
    start("void bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("doFreeze_()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceLine(o->loc);
    line("super_type_::doFreeze_();");
    genTraceLine(o->loc);
    line("self.freeze();");
    if (!o->returnType->unwrap()->isValue()) {
      genTraceLine(o->loc);
      line("value_.freeze();");
    }
    for (auto o : params) {
      if (!o->type->isValue()) {
        genTraceLine(o->loc);
        line(o->name << ".freeze();");
      }
    }
    for (auto o : locals) {
      if (!o->type->isValue()) {
        genTraceLine(o->loc);
        line(getName(o->name->str(), o->number) << ".freeze();");
      }
    }
    out();
    line("}\n");
  }

  /* thaw function */
  if (header) {
    start("virtual void ");
  } else {
    genTraceLine(o->loc);
    start("void bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("doThaw_(libbirch::Label* label_)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceLine(o->loc);
    line("super_type_::doThaw_(label_);");
    genTraceLine(o->loc);
    line("self.thaw(label_);");
    if (!o->returnType->unwrap()->isValue()) {
      genTraceLine(o->loc);
      line("value_.thaw(label_);");
    }
    for (auto o : params) {
      if (!o->type->isValue()) {
        genTraceLine(o->loc);
        line(o->name << ".thaw(label_);");
      }
    }
    for (auto o : locals) {
      if (!o->type->isValue()) {
        genTraceLine(o->loc);
        line(getName(o->name->str(), o->number) << ".thaw(label_);");
      }
    }
    out();
    line("}\n");
  }

  /* finish function */
  if (header) {
    start("virtual void ");
  } else {
    genTraceLine(o->loc);
    start("void bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("doFinish_()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceLine(o->loc);
    line("super_type_::doFinish_();");
    genTraceLine(o->loc);
    line("self.finish();");
    if (!o->returnType->unwrap()->isValue()) {
      genTraceLine(o->loc);
      line("value_.finish();");
    }
    for (auto o : params) {
      if (!o->type->isValue()) {
        genTraceLine(o->loc);
        line(o->name << ".finish();");
      }
    }
    for (auto o : locals) {
      if (!o->type->isValue()) {
        genTraceLine(o->loc);
        line(getName(o->name->str(), o->number) << ".finish();");
      }
    }
    out();
    line("}");
  }

  /* query function */
  if (header) {
    start("virtual ");
  } else {
    genTraceLine(o->loc);
    start("");
  }
  middle("bool ");
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("query()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceLine(o->loc);
    line("libbirch_swap_context_");
    genTraceLine(o->loc);
    line("libbirch_declare_local_");
    genTraceLine(o->loc);
    genTraceFunction(o->name->str(), o->loc);

    genTraceLine(o->loc);
    line("switch (local->point_) {");
    in();
    for (int s = 0; s <= yields.size(); ++s) {
      genTraceLine(o->loc);
      line("case " << s << ": goto POINT" << s << "_;");
    }
    genTraceLine(o->loc);
    line("default: goto END_;");
    out();
    genTraceLine(o->loc);
    line('}');
    genTraceLine(o->loc);
    line("POINT0_:");
    ++point;

    *this << o->braces->strip();

    genTraceLine(o->loc);
    line("END_:");
    genTraceLine(o->loc);
    line("local->point_ = " << (yields.size() + 1) << ';');
    genTraceLine(o->loc);
    line("return false;");

    out();
    finish("}\n");
  }
  if (header) {
    out();
    line("};\n");
  }

  /* initialisation function */
  if (header) {
    start("virtual ");
  } else {
    genTraceLine(o->loc);
    start("");
  }
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::");
  }
  middle(o->name << '(' << o->params << ')');
  if (header) {
    if (o->has(FINAL)) {
      middle(" final");
    } else if (o->has(ABSTRACT)) {
      middle(" = 0");
    }
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceLine(o->loc);
    line("libbirch_swap_context_");
    genTraceLine(o->loc);
    line("libbirch_declare_self_");
    genTraceLine(o->loc);
    start("return libbirch::make_fiber<" << stateName << ">(context_, self");
    for (auto param: params) {
      middle(", " << param->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}

void bi::CppMemberFiberGenerator::visit(const This* o) {
  middle("local->self");
}

void bi::CppMemberFiberGenerator::visit(const Super* o) {
  middle("local->self");
}
