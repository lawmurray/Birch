/**
 * @file
 */
#include "src/generate/CppGenerator.hpp"

#include "src/generate/CppClassGenerator.hpp"
#include "src/primitive/encode.hpp"

birch::CppGenerator::CppGenerator(std::ostream& base, const int level,
    const bool header, const bool generic) :
    IndentableGenerator(base, level),
    header(header),
    generic(generic),
    inAssign(0),
    inGlobal(0),
    inConstructor(0),
    inOperator(0),
    inLambda(0),
    inMember(0),
    inSequence(0),
    inReturn(0) {
  //
}

void birch::CppGenerator::visit(const Name* o) {
  middle(internalise(o->str()));
}

void birch::CppGenerator::visit(const ExpressionList* o) {
  middle(o->head);
  if (o->tail) {
    middle(", " << o->tail);
  }
}

void birch::CppGenerator::visit(const Literal<bool>* o) {
  middle(o->str);
}

void birch::CppGenerator::visit(const Literal<int64_t>* o) {
  middle("birch::type::Integer(" << o->str << ')');
}

void birch::CppGenerator::visit(const Literal<double>* o) {
  middle(o->str);
}

void birch::CppGenerator::visit(const Literal<const char*>* o) {
  middle("birch::type::String(" << o->str << ')');
}

void birch::CppGenerator::visit(const Parentheses* o) {
  auto stripped = o->strip();
  if (stripped->isTuple()) {
    if (inReturn) {
      middle("libbirch::make_tuple");
    } else {
      middle("libbirch::tie");
    }
  }
  middle('(' << stripped << ')');
}

void birch::CppGenerator::visit(const Sequence* o) {
  if (o->single->isEmpty()) {
    middle("libbirch::nil");
  } else if (!inSequence) {
    middle("libbirch::make_array_from_sequence(");
    ++inSequence;
    middle("{ " << o->single << " }");
    --inSequence;
    middle(')');
  } else {
    middle("{ " << o->single << " }");
  }
}

void birch::CppGenerator::visit(const Cast* o) {
  middle("libbirch::cast<" << o->returnType << ">(" << o->single << ')');
}

void birch::CppGenerator::visit(const Call* o) {
  middle(o->single << '(' << o->args);
  if (!o->args->isEmpty()) {
    middle(", ");
  }
  if (inOperator || inGlobal) {
    /* in these contexts, there is no handler to pass on, provide nullptr */
    middle("nullptr");
  } else {
    middle("handler_");
  }
  middle(')');
}

void birch::CppGenerator::visit(const BinaryCall* o) {
  if (isTranslatable(o->name->str())) {
    middle(o->left << ' ' << o->name->str() << ' ' << o->right);
  } else {
    middle(o->name << '(' << o->left << ", " << o->right << ')');
  }
}

void birch::CppGenerator::visit(const UnaryCall* o) {
  if (isTranslatable(o->name->str())) {
    middle(o->name->str() << o->single);
  } else {
    middle(o->name << '(' << o->single << ')');
  }
}

void birch::CppGenerator::visit(const Assign* o) {
  if (o->left->isSlice()) {
    auto slice = dynamic_cast<const Slice*>(o->left);
    middle(slice->single << ".set");
    middle("(libbirch::make_slice(" << slice->brackets << "), " << o->right << ')');
  } else {
    if (o->left->isMembership()) {
      ++inAssign;
    }
    middle(o->left << " = " << o->right);
  }
}

void birch::CppGenerator::visit(const Slice* o) {
  middle(o->single << ".get");
  middle("(libbirch::make_slice(" << o->brackets << "))");
}

void birch::CppGenerator::visit(const Query* o) {
  middle(o->single << ".query()");
}

void birch::CppGenerator::visit(const Get* o) {
  middle(o->single << ".get()");
}

void birch::CppGenerator::visit(const LambdaFunction* o) {
  middle("std::function<" << o->returnType << '(');
  for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
    auto param = dynamic_cast<const Parameter*>(*iter);
    if (iter != o->params->begin()) {
      middle(',');
    }
    middle(param->type);
  }
  if (!o->params->isEmpty()) {
    middle(',');
  }
  finish("const libbirch::Lazy<libbirch::Shared<birch::type::Handler>>&");
  middle(")>([=](" << o->params);
  if (!o->params->isEmpty()) {
    middle(", ");
  }
  finish("const libbirch::Lazy<libbirch::Shared<birch::type::Handler>>& handler_) {");
  in();
  ++inLambda;
  *this << o->braces->strip();
  --inLambda;
  out();
  start("})");
}

void birch::CppGenerator::visit(const Span* o) {
  if (o->single->isEmpty()) {
    middle('0');
  } else {
    middle(o->single);
  }
}

void birch::CppGenerator::visit(const Index* o) {
  middle(o->single << " - 1");
}

void birch::CppGenerator::visit(const Range* o) {
  middle("libbirch::make_range(" << o->left << " - 1, " << o->right << " - 1)");
}

void birch::CppGenerator::visit(const Member* o) {
  if (inConstructor && (o->left->isThis() || o->left->isSuper())) {
    /* don't have access to the `self` variable here, but because we're
     * constructing, so the object cannot possibly be frozen, `this`
     * suffices */
    if (o->left->isThis()) {
      middle("this->");
    } else if (o->left->isSuper()) {
      middle("this->super_type_::");
    }
  } else {
    if (o->left->isThis()) {
      middle("this_()->");
    } else if (o->left->isSuper()) {
      middle("this_()->super_type_::");
    } else {
      middle(o->left);
      auto named = dynamic_cast<const NamedExpression*>(o->right);
      assert(named);
      if (!inAssign && named->category == MEMBER_VARIABLE &&
          named->type->isValue()) {
        /* read optimization: just reading a value, no need to copy-on-write
         * the containing object */
        middle(".read()");
      }
	    inAssign = 0;
      middle("->");
    }
  }
  ++inMember;
  middle(o->right);
  --inMember;
}

void birch::CppGenerator::visit(const This* o) {
  if (inConstructor) {
    middle("this_()");
  } else {
    middle("shared_from_this_()");
  }
}

void birch::CppGenerator::visit(const Super* o) {
  assert(false);  // should be handled by visit(Member*)
}

void birch::CppGenerator::visit(const Global* o) {
  middle(o->single);
}

void birch::CppGenerator::visit(const Nil* o) {
  middle("libbirch::nil");
}

void birch::CppGenerator::visit(const Parameter* o) {
  middle("const " << o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void birch::CppGenerator::visit(const NamedExpression* o) {
  if (o->isGlobal()) {
    middle("birch::" << o->name);
    if (o->category == GLOBAL_VARIABLE) {
      /* global variables generated as functions */
      middle("()");
    }
  } else if (o->isMember()) {
    if (!inMember && !inConstructor) {
      middle("this_()->");
    }
    middle(o->name);
  } else {
    middle(o->name);
  }
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void birch::CppGenerator::visit(const File* o) {
  *this << o->root;
}

void birch::CppGenerator::visit(const GlobalVariable* o) {
  /* C++ does not guarantee static initialization order across compilation
   * units. Global variables are therefore used through accessor functions
   * that initialize their values on first use. */
  ++inGlobal;
  genSourceLine(o->loc);
  start(o->type << "& ");
  if (!header) {
    middle("birch::");
  }
  middle(o->name << "()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genSourceLine(o->loc);
    start("static " << o->type << " result = ");
    genInit(o);
    finish(';');
    genSourceLine(o->loc);
    line("return result;");
    out();
    line("}\n");
  }
  --inGlobal;
}

void birch::CppGenerator::visit(const MemberVariable* o) {
  assert(false);  // should be in CppClassGenerator
}

void birch::CppGenerator::visit(const LocalVariable* o) {
  genTraceLine(o->loc);
  if (o->has(LET)) {
    start("auto " << o->name);
  } else {
    start(o->type << ' ' << o->name);
  }
  middle(" = ");
  genInit(o);
  finish(';');
}

void birch::CppGenerator::visit(const Function* o) {
  if ((generic || !o->isGeneric()) && !o->braces->isEmpty()) {
    genTemplateParams(o);
    genSourceLine(o->loc);
    start(o->returnType << ' ');
    if (!header) {
      middle("birch::");
    }
    middle(o->name << '(' << o->params);
    if (!o->params->isEmpty()) {
      middle(", ");
    }
    middle("const libbirch::Lazy<libbirch::Shared<birch::type::Handler>>& handler_)");
    if (header) {
      finish(";");
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      *this << o->braces->strip();
      out();
      line("}\n");
    }
  }
}

void birch::CppGenerator::visit(const MemberFunction* o) {
  assert(false);  // should be in CppClassGenerator
}

void birch::CppGenerator::visit(const Program* o) {
  if (!o->braces->isEmpty()) {
    genSourceLine(o->loc);
    if (header) {
      line("extern \"C\" int " << o->name << "(int, char**);");
    } else {
      line("int birch::" << o->name << "(int argc_, char** argv_) {");
      in();
      genTraceFunction(o->name->str(), o->loc);

      /* default handler */
      line("libbirch::Lazy<libbirch::Shared<birch::type::PlayHandler>> handler_(true, nullptr);\n");

      /* program options */
      if (o->params->width() > 0) {
        /* option variables */
        for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
          auto param = dynamic_cast<const Parameter*>(*iter);
          assert(param);
          genSourceLine(o->loc);
          start(param->type << ' ' << param->name);
          if (!param->value->isEmpty()) {
            middle(" = " << param->value);
          } else if (param->type->isClass()) {
            middle(" = libbirch::make<" << param->type << ">(handler_)");
          }
          finish(';');
        }
        line("");

        /* option flags */
        line("enum {");
        in();
        for (auto param : *o->params) {
          auto name = dynamic_cast<const Parameter*>(param)->name;
          auto flag = internalise(name->str()) + "FLAG_";
          line(flag << ',');
        }
        out();
        line("};");

        /* long options */
        genSourceLine(o->loc);
        line("int c_, option_index_;");
        genSourceLine(o->loc);
        line("option long_options_[] = {");
        in();
        for (auto param : *o->params) {
          auto name = dynamic_cast<const Parameter*>(param)->name;
          auto flag = internalise(name->str()) + "FLAG_";
          auto option = name->str();
          boost::replace_all(option, "_", "-");

          genSourceLine(o->loc);
          start("{\"");
          middle(option << "\", required_argument, 0, " << flag);
          finish(" },");
        }
        genSourceLine(o->loc);
        line("{0, 0, 0, 0}");
        out();
        genSourceLine(o->loc);
        line("};");

        /* short options */
        genSourceLine(o->loc);
        line("const char* short_options_ = \":\";");

        /* handle error reporting ourselves (worth commenting this out if
        * debugging issues with command-line parsing) */
        genSourceLine(o->loc);
        line("::opterr = 0;");

        /* read in options with getopt_long */
        genSourceLine(o->loc);
        start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
        finish("long_options_, &option_index_);");
        genSourceLine(o->loc);
        line("while (c_ != -1) {");
        in();
        genSourceLine(o->loc);
        line("switch (c_) {");
        in();

        for (auto param : *o->params) {
          auto p = dynamic_cast<const Parameter*>(param);
          auto name = p->name;
          auto flag = internalise(name->str()) + "FLAG_";

          genSourceLine(p->loc);
          line("case " << flag << ':');
          in();
          genSourceLine(p->loc);
          line("libbirch_error_msg_(::optarg, \"option --\" << long_options_[::optopt].name << \" requires a value.\");");
          genSourceLine(p->loc);
          if (p->type->unwrap()->isBasic()) {
            auto type = dynamic_cast<Named*>(p->type->unwrap());
            assert(type);
            start(name << " = birch::" << type->name);
            finish("(birch::type::String(::optarg), handler_);");
          } else {
            line(name << " = birch::type::String(::optarg);");
          }
          line("break;");
          out();
        }

        genSourceLine(o->loc);
        line("case '?':");
        in();
        genSourceLine(o->loc);
        line("libbirch_error_msg_(false, \"option \" << argv_[::optind - 1] << \" unrecognized.\");");
        out();

        genSourceLine(o->loc);
        line("case ':':");
        in();
        genSourceLine(o->loc);
        line("libbirch_error_msg_(false, \"option --\" << long_options_[::optopt].name << \" requires a value.\");");
        out();

        genSourceLine(o->loc);
        line("default:");
        in();
        genSourceLine(o->loc);
        line("libbirch_error_msg_(false, std::string(\"unknown error parsing command-line options.\"));");
        out();

        out();
        line('}');
        genSourceLine(o->loc);
        start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
        finish("long_options_, &option_index_);");
        out();
        line("}\n");
      }

      /* body of program */
      *this << o->braces->strip();

      genTraceLine(o->loc);
      line("return 0;");
      out();
      line("}\n");
    }
  }
}

void birch::CppGenerator::visit(const BinaryOperator* o) {
  if (!o->braces->isEmpty()) {
    genSourceLine(o->loc);
    start(o->returnType << ' ');
    if (!header) {
      middle("birch::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->left << ", " << o->right << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      ++inOperator;
      *this << o->braces->strip();
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppGenerator::visit(const UnaryOperator* o) {
  if (!o->braces->isEmpty()) {
    genSourceLine(o->loc);
    start(o->returnType << ' ');
    if (!header) {
      middle("birch::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      ++inOperator;
      *this << o->braces->strip();
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppGenerator::visit(const AssignmentOperator* o) {
  assert(false);  // should be in CppClassGenerator
}

void birch::CppGenerator::visit(const ConversionOperator* o) {
  assert(false);  // should be in CppClassGenerator
}

void birch::CppGenerator::visit(const Basic* o) {
  //
}

void birch::CppGenerator::visit(const Class* o) {
  if (generic || !o->isGeneric()) {
    CppClassGenerator auxClass(base, level, header, generic);
    auxClass << o;
  }
}

void birch::CppGenerator::visit(const Generic* o) {
  middle(o->name);
}

void birch::CppGenerator::visit(const Braces* o) {
  line('{');
  in();
  *this << o->single;
  out();
  line('}');
}

void birch::CppGenerator::visit(const Assume* o) {
  genTraceLine(o->loc);
  if (*o->name == "<-?") {
    line("libbirch::optional_assign(" << o->left << ", " << o->right << ");");
  } else if (*o->name == "<~") {
    start("libbirch::simulate(" << o->left << ", birch::SimulateEvent(");
    finish('(' << o->right << ")->distribution(handler_), handler_), handler_);");
  } else if (*o->name == "~>") {
    start("libbirch::observe(birch::ObserveEvent(" << o->left << ", ");
    finish('(' << o->right << ")->distribution(handler_), handler_), handler_);");
  } else if (*o->name == "~") {
    start("libbirch::assume(birch::AssumeEvent(" << o->left << ", ");
    finish('(' << o->right << ")->distribution(handler_), handler_), handler_);");
  } else {
    assert(false);
  }
}

void birch::CppGenerator::visit(const Factor* o) {
  genTraceLine(o->loc);
  start("libbirch::factor(birch::FactorEvent(" << o->single);
  finish(", handler_), handler_);");
}

void birch::CppGenerator::visit(const ExpressionStatement* o) {
  genTraceLine(o->loc);
  line(o->single << ';');
}

void birch::CppGenerator::visit(const If* o) {
  genTraceLine(o->loc);
  line("if (" << o->cond->strip() << ") {");
  in();
  *this << o->braces->strip();
  out();
  if (!o->falseBraces->isEmpty()) {
    line("} else {");
    in();
    *this << o->falseBraces->strip();
    out();
  }
  line("}");
}

void birch::CppGenerator::visit(const For* o) {
  auto index = getIndex(o->index);
  genTraceLine(o->loc);
  start("for (auto " << index << " = " << o->from << "; ");
  finish(index << " <= " << o->to << "; ++" << index << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void birch::CppGenerator::visit(const Parallel* o) {
  auto index = getIndex(o->index);
  genTraceLine(o->loc);
  line("#pragma omp parallel");
  line("{");
  in();
  genTraceFunction("<parallel for>", o->loc);
  start("#pragma omp for schedule(");
  if (o->has(DYNAMIC)) {
    middle("guided");
  } else {
    middle("static");
  }
  finish(')');
  start("for (auto " << index << " = " << o->from << "; ");
  finish(index << " <= " << o->to << "; ++" << index << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
  out();
  line("}");
}

void birch::CppGenerator::visit(const While* o) {
  genTraceLine(o->loc);
  line("while (" << o->cond->strip() << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void birch::CppGenerator::visit(const DoWhile* o) {
  genTraceLine(o->loc);
  line("do {");
  in();
  *this << o->braces->strip();
  out();
  line("} while (" << o->cond->strip() << ");");
}

void birch::CppGenerator::visit(const With* o) {
  genTraceLine(o->loc);
  line("{");
  in();
  line("auto handler_ = " << o->single << ';');
  *this << o->braces->strip();
  out();
  line("}");
}

void birch::CppGenerator::visit(const Assert* o) {
  genTraceLine(o->loc);
  line("libbirch_assert_(" << o->cond->strip() << ");");
}

void birch::CppGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  ++inReturn;
  line("return " << o->single << ';');
  --inReturn;
}

void birch::CppGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    if (!header) {
      genSourceLine(o->loc);
    }
    *this << escape_unicode(o->raw);
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void birch::CppGenerator::visit(const StatementList* o) {
  middle(o->head << o->tail);
}

void birch::CppGenerator::visit(const EmptyType* o) {
  middle("void");
}

void birch::CppGenerator::visit(const ArrayType* o) {
  middle("libbirch::DefaultArray<" << o->single << ',' << o->depth() << '>');
}

void birch::CppGenerator::visit(const TupleType* o) {
  middle("libbirch::Tuple<" << o->single << '>');
}

void birch::CppGenerator::visit(const FunctionType* o) {
  middle("std::function<" << o->returnType << '(' << o->params);
  if (!o->params->isEmpty()) {
    middle(", ");
  }
  middle("const libbirch::Lazy<libbirch::Shared<birch::type::Handler>>&)>");
}

void birch::CppGenerator::visit(const OptionalType* o) {
  middle("libbirch::Optional<" << o->single << '>');
}

void birch::CppGenerator::visit(const MemberType* o) {
  middle(o->right);
}

void birch::CppGenerator::visit(const NamedType* o) {
  if (o->isClass()) {
    middle("libbirch::Lazy<libbirch::Shared<");
    middle("birch::type::" << o->name);
    if (!o->typeArgs->isEmpty()) {
      middle('<' << o->typeArgs << '>');
    }
    middle(">>");
  } else if (o->isBasic()) {
    middle("birch::type::" << o->name);
    if (!o->typeArgs->isEmpty()) {
      middle('<' << o->typeArgs << '>');
    }
  } else if (o->isGeneric()) {
    middle(o->name);
    if (!o->typeArgs->isEmpty()) {
      middle('<' << o->typeArgs << '>');
    }
  } else {
    assert(false);
  }
}

void birch::CppGenerator::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

std::string birch::CppGenerator::getIndex(const Statement* o) {
  auto index = dynamic_cast<const LocalVariable*>(o);
  assert(index);
  return internalise(index->name->str());
}

void birch::CppGenerator::genTraceFunction(const std::string& name,
    const Location* loc) {
  genSourceLine(loc);
  start("libbirch_function_(\"" << name << "\", \"");
  finish(loc->file->path << "\", " << loc->firstLine << ");");
}

void birch::CppGenerator::genTraceLine(const Location* loc) {
  genSourceLine(loc);
  line("libbirch_line_(" << loc->firstLine << ");");
  genSourceLine(loc);
}

void birch::CppGenerator::genSourceLine(const Location* loc) {
  auto line = loc->firstLine;
  auto file = loc->file->path;
  line("#line " << line << " \"" << file << "\"");
}
