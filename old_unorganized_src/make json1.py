import ast
import json
import typing
from collections import defaultdict

def parse_expression_ast(expression: str) -> typing.Dict:
    """
    Parse a string expression into an Abstract Syntax Tree representation.
    Terminal nodes (variables, constants, function names as arguments) are marked with '_'.
    
    Args:
        expression (str): A string containing a function call expression
    
    Returns:
        dict: A dictionary representing the parsed expression tree
    """
    class ExpressionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.tree = defaultdict(list)
            self.current_nested_path = []
            self.current_args = False  # Flag to track if we're processing arguments
        
        def visit_Call(self, node: ast.Call):
            # Extract function name
            func_name = self._extract_function_name(node.func)
            
            # Process arguments
            arg_names = []
            old_args_state = self.current_args  # Save the current state
            self.current_args = True  # We're now processing arguments
            
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    # Recursively process nested function calls
                    self.current_nested_path.append(func_name)
                    self.visit(arg)
                    arg_name = self._extract_node_info(arg)
                    arg_names.append(arg_name)  # Don't mark nested calls
                elif isinstance(arg, ast.Name):
                    # Mark any name when it appears as an argument with underscore
                    arg_name = f"_{arg.id}"  # Always mark with underscore when used as argument
                    arg_names.append(arg_name)
                elif isinstance(arg, ast.Constant):
                    # Mark constants with underscore
                    arg_name = f"_{str(arg.value)}"
                    arg_names.append(arg_name)
            
            self.current_args = old_args_state  # Restore the previous state
            
            # Store the function call in the tree
            if arg_names:
                self.tree[func_name].append(arg_names)
            
            # If we've processed nested calls, pop the current function from the path
            if self.current_nested_path and self.current_nested_path[-1] == func_name:
                self.current_nested_path.pop()
            
            return node
        
        def _extract_function_name(self, func_node):
            """
            Extract the function or variable name
            """
            if isinstance(func_node, ast.Name):
                return func_node.id
            elif isinstance(func_node, ast.Attribute):
                return func_node.attr
            return func_node.__class__.__name__
        
        def _extract_node_info(self, node: ast.Call):
            """
            Extract the name of the function or variable
            """
            return self._extract_function_name(node.func)
        
        def visit_Name(self, node: ast.Name):
            """
            Handle standalone variable names
            """
            # Only add if it's not part of a function call and we're not in args
            if not self.current_nested_path and not self.current_args:
                # Mark standalone variables with underscore
                self.tree[node.id].append([f"_{node.id}"])
            return node
        
        def generic_visit(self, node):
            # Continue traversing the tree for other node types
            super().generic_visit(node)
    
    def create_expression(expression: str) -> str:
        """
        Attempt to convert the expression to a valid Python expression.
        """
        attempts = [
            f"lambda I: {expression}",
            expression,
            f"({expression})",
            f"lambda: {expression}",
        ]
        
        for attempt in attempts:
            try:
                parsed = ast.parse(attempt)
                return attempt
            except SyntaxError:
                continue
        
        raise ValueError(f"Unable to parse expression: {expression}")
    
    try:
        valid_expr = create_expression(expression)
        parsed = ast.parse(valid_expr)
        visitor = ExpressionVisitor()
        visitor.visit(parsed)
        
        first_func = None
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                first_func = visitor._extract_function_name(node.func)
                break
        
        if first_func:
            visitor.tree['root'] = [[first_func]]
        
        return dict(visitor.tree)
    
    except ValueError as ve:
        raise ValueError(f"Failed to parse expression '{expression}': {ve}")
    
def parse_functional_expressions(input_file):
    """
    Parse functional expressions using the AST parser
    
    Args:
        input_file (str): Path to the input file
    
    Returns:
        dict: Comprehensive node tree of function relationships
    """
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Dictionary to store all function definitions
    func_definitions = {}
    
    # Execute the content to get function definitions
    exec_globals = {}
    exec(content, exec_globals)
    
    # Comprehensive node dictionary
    comprehensive_tree = defaultdict(list)
    # Keep track of all top-level functions
    root_functions = list()
    
    # Process each verify function
    for func_name, func_def in exec_globals.items():
        if isinstance(func_def, str) and func_name.startswith('verify_'):
            try:
                # Parse the functional expression
                tree_dict = parse_expression_ast(func_def)
                
                # Store the root function for this expression
                #if 'root' in tree_dict:
                #    for root_list in tree_dict['root']:
                #        root_functions.extend(root_list)
                
                # Merge the parsed tree into comprehensive tree
                for node, children_lists in tree_dict.items():
                    #if node != 'root':  # Skip the individual root nodes
                    comprehensive_tree[node].extend(children_lists)
                    # Remove duplicates while preserving order for each function
                    #comprehensive_tree[node] = [
                    #    list(t) for t in {tuple(x) for x in comprehensive_tree[node]}
                    #]
            
            except ValueError as e:
                print(f"Error parsing {func_name}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {func_name}: {e}")
    
    # Add the comprehensive root node with all possible root functions
    #comprehensive_tree['root'] = [[f] for f in sorted(root_functions)]
    
    # Convert to regular dict
    comprehensive_tree = dict(comprehensive_tree)
    
    return comprehensive_tree

def parse_and_save_tree(input_file, output_file):
    """
    Parse functional expressions and save the node tree to a JSON file
    
    Args:
        input_file (str): Input Python file with functional expressions
        output_file (str): Output JSON file to save the node tree
    """
    # Parse the expressions
    comprehensive_tree = parse_functional_expressions(input_file)
    
    # Write to JSON
    with open(output_file, 'w') as f:
        json.dump(comprehensive_tree, f, indent=2)
    
    print(f"Comprehensive node tree saved to {output_file}")

def main():
    input_file = 'functional_dsl_functions.py'
    output_file = 'node_tree.json'
    
    parse_and_save_tree(input_file, output_file)

if __name__ == "__main__":
    main()