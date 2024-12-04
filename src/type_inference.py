import ast
import inspect
from types import ModuleType
from typing import Dict, Any, List
import importlib.util
import sys

class TypeTracer:
    def __init__(self):
        self.traces = {}
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Record argument types
            arg_types = [type(arg).__name__ for arg in args]
            func_name = func.__name__
            
            # Call function
            result = func(*args, **kwargs)
            
            # Record return type
            return_type = type(result).__name__
            
            # Store trace
            if func_name not in self.traces:
                self.traces[func_name] = []
            self.traces[func_name].append({
                'args': arg_types,
                'return': return_type
            })
            
            return result
        return wrapper

def load_dsl_with_tracing():
    # Load DSL module
    spec = importlib.util.spec_from_file_location("dsl", "dsl.py")
    dsl = importlib.util.module_from_spec(spec)
    sys.modules["dsl"] = dsl
    spec.loader.exec_module(dsl)
    
    # Create tracer
    tracer = TypeTracer()
    
    # Wrap DSL functions with tracer
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        setattr(dsl, name, tracer(func))
    
    return dsl, tracer

def analyze_types(expr: str):
    dsl, tracer = load_dsl_with_tracing()
    eval(expr, dsl.__dict__)
    return tracer.traces

# Test
expr = "combine(initset(increment(5)), initset(3))"
print(analyze_types(expr))