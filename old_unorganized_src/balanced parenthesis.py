import re

def check_parenthesis_balance(input_file):
    """
    Check all expressions in the input file for balanced parentheses.
    
    Args:
        input_file (str): Path to the input file containing functional DSL functions
    
    Returns:
        tuple: (is_all_balanced, unbalanced_functions)
    """
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Execute the content to get function definitions
    exec_globals = {}
    exec(content, exec_globals)
    
    # Track unbalanced functions
    unbalanced_functions = []
    
    def is_balanced(expression):
        """
        Check if parentheses in the expression are balanced.
        
        Args:
            expression (str): Expression to check
        
        Returns:
            bool: True if parentheses are balanced, False otherwise
        """
        # Remove non-parenthesis characters to focus on balance
        paren_only = re.sub(r'[^()]+', '', expression)
        
        # Use stack to track parenthesis balance
        stack = []
        for char in paren_only:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        
        # Expression is balanced if stack is empty
        return len(stack) == 0

    def extract_verify_functions(exec_globals):
        """
        Extract all verify functions from the executed globals.
        
        Args:
            exec_globals (dict): Globals from executed file
        
        Returns:
            list: List of (function_name, function_expression) tuples
        """
        return [
            (func_name, func_def) 
            for func_name, func_def in exec_globals.items() 
            if isinstance(func_def, str) and func_name.startswith('verify_')
        ]

    # Check balance for each verify function
    all_balanced = True
    for func_name, func_def in extract_verify_functions(exec_globals):
        # Trim any extra whitespace
        func_def = func_def.strip()
        
        # Check balance
        if not is_balanced(func_def):
            unbalanced_functions.append((func_name, func_def))
            all_balanced = False
    
    # Detailed reporting
    if all_balanced:
        print("✅ All expressions have balanced parentheses!")
    else:
        print("❌ Some expressions have unbalanced parentheses:")
        for func_name, func_def in unbalanced_functions:
            print(f"\nUnbalanced Function: {func_name}")
            print(f"Expression: {func_def}")
            
            # Diagnostic information
            open_count = func_def.count('(')
            close_count = func_def.count(')')
            print(f"Open Parentheses: {open_count}")
            print(f"Close Parentheses: {close_count}")
    
    return all_balanced, unbalanced_functions

def main():
    input_file = 'functional_dsl_functions.py'
    
    # Run balance check
    is_balanced, unbalanced = check_parenthesis_balance(input_file)
    
    # Exit with appropriate status code
    if not is_balanced:
        print("\n❗ Parenthesis balance check FAILED!")
        exit(1)
    
    print("\n✅ Parenthesis balance check PASSED!")
    exit(0)

if __name__ == "__main__":
    main()