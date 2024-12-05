import re

def parse_dsl_functions(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find all function definitions
    function_pattern = re.compile(r'def\s+(verify_[a-f0-9]+)\(I:\s*Grid\)\s*->\s*Grid:\s*((?:.*\n)*?)\s*return\s*(\w+)', re.MULTILINE)
    
    # Output dictionary to store parsed functions
    parsed_functions = {}
    
    # Iterate through function matches
    for match in function_pattern.finditer(content):
        func_name = match.group(1)
        func_body = match.group(2).strip().split('\n')
        return_var = match.group(3)
        
        # Reconstruct the function body with return statement
        full_body = '\n'.join(func_body + [f'return {return_var}'])
        
        # Store in dictionary
        parsed_functions[func_name] = full_body
    
    # Write to output file
    with open(output_file, 'w') as f:
        for func_name, func_body in parsed_functions.items():
            f.write(f'{func_name} = """\n{func_body}\n"""\n\n')
    
    print(f"Parsed {len(parsed_functions)} functions to {output_file}")

# Usage example
parse_dsl_functions('verifiers.py', 'dsl_functions.py')