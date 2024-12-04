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

# Test the function with the specific example
dsl_function = """
x0 = palette(I)
x1 = other(x0, ZERO)
x2 = shape(I)
x3 = multiply(x2, x2)
x4 = canvas(ZERO, x3)
x5 = ofcolor(I, x1)
x6 = lbind(shift, x5)
x7 = shape(I)
x8 = rbind(multiply, x7)
x9 = apply(x8, x5)
x10 = mapply(x6, x9)
x11 = fill(x4, x1, x10)
return x11
"""

# Transform the function
transformed_func = functional_transform(dsl_function)
print(transformed_func)