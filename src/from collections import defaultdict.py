from collections import defaultdict

# Function to parse an expression into a tree structure
def parse_expression(expression):
    """
    Parse an expression into a tree structure, returning a dictionary
    where each node maps to a list of lists of its children.
    """
    tree = defaultdict(list)  # Dictionary to store nodes and their multiple occurrences

    def helper(expr):
        # Base case: If it's a single node (no parentheses), return it as a leaf
        if isinstance(expr, str) and '(' not in expr:
            return expr.strip()
        
        # Extract the function name and its arguments
        func_name, _, rest = expr.partition('(')
        func_name = func_name.strip()
        arguments = []
        balance = 0
        current_arg = []

        # Split the arguments by commas, taking nested parentheses into account
        for char in rest[:-1]:  # Exclude the closing ')'
            if char == ',' and balance == 0:
                arguments.append(''.join(current_arg).strip())
                current_arg = []
            else:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                current_arg.append(char)
        if current_arg:
            arguments.append(''.join(current_arg).strip())
        
        # Add the function name and its children to the tree
        children = [helper(arg) for arg in arguments]
        tree[func_name].append(children)  # Append the children as a list to account for multiple occurrences
        return func_name

    root = helper(expression)
    tree['root'] = [[root]]
    return root, dict(tree)

# Input expression
expression = "fill(canvas(ZERO, multiply(shape(I), shape(I))), other(palette(I), ZERO), mapply(lbind(shift, ofcolor(I, other(palette(I), ZERO))), apply(rbind(multiply, shape(I)), ofcolor(I, other(palette(I), ZERO)))))"

# Parse the expression
root_node, tree_dict = parse_expression(expression)

# Output the tree dictionary
print("Root Node:", root_node)
print("Tree Dictionary:")
for node, children_lists in tree_dict.items():
    print(f"{node}: {children_lists}")
