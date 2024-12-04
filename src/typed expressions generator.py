import random
from typing import Dict, List, Tuple, Any, Set
import ast

class DSLExpressionGenerator:
    def __init__(self, stats: Dict[Tuple[str, str], List[List[Tuple[Any, str]]]]):
        self.stats = stats
        # Parse type signatures to build a graph of possible function compositions
        self.type_graph = {}  # Maps types to functions that return that type
        self.func_inputs = {}  # Maps functions to their input type requirements
        
        for (func_name, type_sig), usages in stats.items():
            # Split on the rightmost -> to handle nested function types
            parts = type_sig.rsplit(' -> ', 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed type signature for {func_name}: {type_sig}")
                continue
                
            inputs_str, return_type = parts
            # Clean up the input string
            inputs_str = inputs_str.strip('()')
            input_types = [t.strip() for t in inputs_str.split(',') if t.strip()]
            
            # Store in type graph
            if return_type not in self.type_graph:
                self.type_graph[return_type] = []
            self.type_graph[return_type].append((func_name, input_types))
            
            # Store input requirements
            self.func_inputs[func_name] = input_types

        print("Type graph:", self.type_graph)
        print("Function inputs:", self.func_inputs)

        # Create sets of available types and constants
        self.base_types = {'int', 'bool', 'Numerical', 'Container', 'FrozenSet[int]'}
        self.constants = {
            'int': list(range(-2, 11)),  # Based on constants in DSL
            'bool': [True, False]
        }

    def can_generate_type(self, target_type: str, depth: int = 3) -> bool:
        """Check if we can generate an expression of the target type"""
        if depth <= 0:
            return False
            
        # Base cases
        if target_type in self.constants:
            return True
            
        # Check if we have functions that return this type
        if target_type in self.type_graph:
            for func, input_types in self.type_graph[target_type]:
                if all(self.can_generate_type(t, depth-1) for t in input_types):
                    return True
                    
        return False

    def generate_of_type(self, target_type: str, depth: int = 3) -> str:
        """Generate a random expression of the target type"""
        if depth <= 0:
            raise ValueError(f"Cannot generate expression of type {target_type} within depth limit")
            
        # Handle base types directly
        if target_type in self.constants:
            return str(random.choice(self.constants[target_type]))
            
        # Find functions that return our target type
        possible_funcs = self.type_graph.get(target_type, [])
        if not possible_funcs:
            raise ValueError(f"No functions found that return type {target_type}")
            
        # Pick a random function and generate its inputs
        func, input_types = random.choice(possible_funcs)
        
        try:
            args = [self.generate_of_type(t, depth-1) for t in input_types]
            return f"{func}({', '.join(args)})"
        except ValueError as e:
            remaining_funcs = [(f, t) for f, t in possible_funcs if f != func]
            if remaining_funcs:
                func, input_types = random.choice(remaining_funcs)
                args = [self.generate_of_type(t, depth-1) for t in input_types]
                return f"{func}({', '.join(args)})"
            raise ValueError(f"Cannot generate valid expression of type {target_type}")

    def generate_random_expression(self, max_depth: int = 3) -> str:
        """Generate a random type-correct expression"""
        # Pick a random return type from our available types
        available_types = list(self.type_graph.keys())
        if not available_types:
            raise ValueError("No types available to generate expressions")
            
        target_type = random.choice(available_types)
        
        try:
            return self.generate_of_type(target_type, max_depth)
        except ValueError:
            # If we fail with this type, try another
            remaining_types = [t for t in available_types if t != target_type]
            if remaining_types:
                target_type = random.choice(remaining_types)
                return self.generate_of_type(target_type, max_depth)
            raise ValueError("Could not generate any valid expression")

def generate_expressions(stats: Dict[Tuple[str, str], List[List[Tuple[Any, str]]]], 
                       num_expressions: int = 5,
                       max_depth: int = 3) -> List[str]:
    """Generate multiple random type-correct DSL expressions"""
    generator = DSLExpressionGenerator(stats)
    expressions = []
    
    for _ in range(num_expressions):
        try:
            expr = generator.generate_random_expression(max_depth)
            expressions.append(expr)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
            
    return expressions

# Example usage:
if __name__ == "__main__":
    # First get stats using the previous analyzer
    from stats import analyze_typed_dsl_expression
    
    # Get some initial stats to work with
    sample_expressions = [
        "combine(initset(increment(5)), initset(3))",
        "matcher(increment, 6)",
        "compose(matcher(increment, 6), double)"
    ]
    
    all_stats = {}
    for expr in sample_expressions:
        stats = analyze_typed_dsl_expression(expr)
        all_stats.update(stats)
    
    print("\nCollected stats:")
    for (func, sig), usages in all_stats.items():
        print(f"{func}: {sig}")
        for usage in usages:
            print(f"  {usage}")
    
    # Generate new expressions
    new_expressions = generate_expressions(all_stats, num_expressions=5)
    print("\nGenerated expressions:")
    for i, expr in enumerate(new_expressions, 1):
        print(f"{i}. {expr}")