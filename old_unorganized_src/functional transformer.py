def functional_transform(dsl_function):
    """
    Transform DSL-style function with sequential assignments 
    into nested functional style.
    """
    # Parse the function body into lines
    lines = [line.strip() for line in dsl_function.split('\n') if line.strip() and not line.strip().startswith('return')]
    
    # Extract the return variable
    return_line = [line for line in dsl_function.split('\n') if line.strip().startswith('return')][0]
    return_var = return_line.split('return')[1].strip()
    
    def nest_expressions(lines):
        # Start with the final expression
        nested = return_var
        
        # Work backwards through the assignments
        for line in reversed(lines):
            # Split the assignment
            var, expr = [part.strip() for part in line.split('=')]
            
            # Replace variable with its expression
            nested = nested.replace(var, f"{expr}")
        
        return nested
    
    return nest_expressions(lines)

def transform_dsl_functions(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Execute the content to get function definitions
    exec_globals = {}
    exec(content, exec_globals)
    
    # Open output file
    with open(output_file, 'w') as out_f:
        # Iterate through functions
        for func_name, func_def in exec_globals.items():
            if isinstance(func_def, str) and func_name.startswith('verify_'):
                # Transform the function
                transformed = functional_transform(func_def)
                
                # Write to output file
                out_f.write(f'{func_name} = """\n{transformed}\n"""\n\n')
    
    print(f"Transformed functions written to {output_file}")

# Usage
transform_dsl_functions('dsl_functions.py', 'functional_dsl_functions.py')