import ast
import typing

def parse_expression_ast(expression: str) -> typing.Dict:
    """
    Parse a string expression into an Abstract Syntax Tree representation.
    
    Args:
        expression (str): A string containing a nested function call expression
    
    Returns:
        dict: A dictionary representing the parsed expression tree
    """
    class ExpressionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.tree = {}
        
        def visit_Call(self, node: ast.Call):
            # Extract function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            else:
                func_name = 'unknown'
            
            # Process arguments
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    # Recursively process nested function calls
                    self.visit(arg)
                    args.append(self._extract_node_info(arg))
                elif isinstance(arg, ast.Name):
                    args.append(arg.id)
                elif isinstance(arg, ast.Constant):
                    args.append(str(arg.value))
            
            # Store the function call in the tree
            if func_name not in self.tree:
                self.tree[func_name] = []
            self.tree[func_name].append(args)
            
            return node
        
        def _extract_node_info(self, node: ast.Call):
            # Helper to extract function name from a node
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
            return 'unknown'
        
        def generic_visit(self, node):
            # Continue traversing the tree for other node types
            super().generic_visit(node)
    
    def create_lambda_expression(expression: str) -> str:
        """
        Convert the expression to a valid Python lambda for AST parsing.
        """
        # Wrap the expression in a lambda to make it a valid Python expression
        return f"lambda I: {expression}"
    
    try:
        # Create a lambda expression to parse
        lambda_expr = create_lambda_expression(expression)
        
        # Parse the lambda expression
        parsed = ast.parse(lambda_expr)
        
        # Create and run the visitor
        visitor = ExpressionVisitor()
        visitor.visit(parsed)
        
        # Add a root key to mimic the original implementation
        root = list(visitor.tree.keys())[0] if visitor.tree else None
        visitor.tree['root'] = [[root]]
        
        return root, dict(visitor.tree)
    
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}")

# Test the parser
if __name__ == "__main__":
    # Example expression
    expression = "fill(canvas(ZERO, multiply(shape(I), shape(I))), other(palette(I), ZERO), mapply(lbind(shift, ofcolor(I, other(palette(I), ZERO))), apply(rbind(multiply, shape(I)), ofcolor(I, other(palette(I), ZERO)))))"
    
    try:
        root_node, tree_dict = parse_expression_ast(expression)
        
        print("Root Node:", root_node)
        print("\nTree Dictionary:")
        for node, children_lists in tree_dict.items():
            print(f"{node}: {children_lists}")
    
    except Exception as e:
        print(f"Error parsing expression: {e}")