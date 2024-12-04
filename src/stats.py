import ast
from typing import Dict, List, Any, Tuple, Optional
import os

class TypeAnnotationExtractor(ast.NodeVisitor):
    def __init__(self):
        self.function_types: Dict[str, Tuple[List[str], str]] = {}
        self.detailed_types: Dict[str, Tuple[List[str], str]] = {}
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        param_types = []
        for arg in node.args.args:
            if arg.annotation:
                param_types.append(ast.unparse(arg.annotation))
            else:
                param_types.append("Any")
        
        return_type = ast.unparse(node.returns) if node.returns else "Any"
        self.function_types[node.name] = (param_types, return_type)
        self.detailed_types[node.name] = (param_types, return_type)

class TypeInferer:
    @staticmethod
    def infer_constant_type(value: Any) -> str:
        if isinstance(value, int):
            return "int"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, tuple):
            return f"Tuple[{', '.join(TypeInferer.infer_constant_type(x) for x in value)}]"
        return type(value).__name__

    @staticmethod
    def infer_function_type(func_name: str, detailed_types: Dict[str, Tuple[List[str], str]]) -> Optional[Tuple[List[str], str]]:
        """Infer input and output types of a function"""
        if func_name in detailed_types:
            return detailed_types[func_name]
        return None

    @staticmethod
    def infer_return_type(func_name: str, arg_types: List[str], detailed_types: Dict[str, Tuple[List[str], str]]) -> str:
        if func_name == "matcher":
            if len(arg_types) == 2:
                # If first argument is a function type signature
                func_type = arg_types[0]
                if "->" in func_type:
                    input_type = func_type.split("->")[0].strip()[1:-1]  # Extract input type
                    return f"({input_type}) -> bool"
        elif func_name == "increment" and arg_types == ["int"]:
            return "int"
        elif func_name == "initset":
            if len(arg_types) == 1:
                return f"FrozenSet[{arg_types[0]}]"
        elif func_name == "combine" and all(t.startswith("FrozenSet[") for t in arg_types):
            element_type = arg_types[0][len("FrozenSet["):-1]
            return f"FrozenSet[{element_type}]"
        return detailed_types.get(func_name, ([], "Any"))[1]

    @staticmethod
    def make_specific_signature(func_name: str, arg_types: List[str], return_type: str) -> str:
        return f"({', '.join(arg_types)}) -> {return_type}"

class TypedDSLAnalyzer(ast.NodeVisitor):
    def __init__(self, type_info: Dict[str, Tuple[List[str], str]], detailed_info: Dict[str, Tuple[List[str], str]]):
        self.function_types = type_info
        self.detailed_types = detailed_info
        self.function_stats: Dict[Tuple[str, str], List[List[Tuple[Any, str]]]] = {}
        
    def get_function_type_str(self, func_name: str) -> str:
        if func_name in self.detailed_types:
            params, ret = self.detailed_types[func_name]
            return f"({', '.join(params)}) -> {ret}"
        return "(?)"
        
    def visit_Call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            typed_args = []
            arg_types = []
            
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    arg_type = TypeInferer.infer_constant_type(arg.value)
                    typed_args.append((arg.value, arg_type))
                    arg_types.append(arg_type)
                elif isinstance(arg, ast.Name):
                    if arg.id in self.detailed_types:
                        func_type = self.get_function_type_str(arg.id)
                        typed_args.append((arg.id, func_type))
                        arg_types.append(func_type)
                    else:
                        typed_args.append((arg.id, "Any"))
                        arg_types.append("Any")
                elif isinstance(arg, ast.Call):
                    inner_return_type = self.visit(arg)
                    inner_func_name = arg.func.id if isinstance(arg.func, ast.Name) else 'complex_call'
                    typed_args.append((f"<{inner_func_name}>", inner_return_type))
                    arg_types.append(inner_return_type)
                else:
                    typed_args.append(("<complex_expression>", "Any"))
                    arg_types.append("Any")
            
            return_type = TypeInferer.infer_return_type(func_name, arg_types, self.detailed_types)
            specific_signature = TypeInferer.make_specific_signature(func_name, arg_types, return_type)
            
            key = (func_name, specific_signature)
            if key not in self.function_stats:
                self.function_stats[key] = []
            self.function_stats[key].append(typed_args)
            
            return return_type
            
        self.generic_visit(node)
        return "Any"

def analyze_typed_dsl_expression(expression: str) -> Dict[Tuple[str, str], List[List[Tuple[Any, str]]]]:
    try:
        with open("dsl.py", 'r') as f:
            dsl_code = f.read()
        
        tree = ast.parse(dsl_code)
        extractor = TypeAnnotationExtractor()
        extractor.visit(tree)
        
        tree = ast.parse(expression)
        analyzer = TypedDSLAnalyzer(extractor.function_types, extractor.detailed_types)
        analyzer.visit(tree)
        return analyzer.function_stats
        
    except Exception as e:
        return {("ERROR", "(?)"): [[(str(e), "str")]]}

def format_typed_stats(stats: Dict[Tuple[str, str], List[List[Tuple[Any, str]]]]) -> str:
    result = []
    for (func_name, type_sig), arg_lists in sorted(stats.items()):
        result.append(f"Function: {func_name}")
        result.append(f"Type signature: {type_sig}")
        for i, args in enumerate(arg_lists, 1):
            args_str = ", ".join(f"{arg[0]}: {arg[1]}" for arg in args)
            result.append(f"  Usage {i}: [{args_str}]")
        result.append("")
    return "\n".join(result)

# Test with higher-order function
expr1 = "matcher(increment, 6)"
print("Higher-order function analysis:")
stats1 = analyze_typed_dsl_expression(expr1)
print(format_typed_stats(stats1))

# Test with composition
expr2 = "compose(matcher(increment, 6), double)"
print("\nComposition analysis:")
stats2 = analyze_typed_dsl_expression(expr2)
print(format_typed_stats(stats2))