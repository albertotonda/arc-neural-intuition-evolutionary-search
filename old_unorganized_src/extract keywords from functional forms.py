def parse_expression(expression):
    """
    Parse nested expressions into a list of unique keywords.
    Skips parentheses and commas, removes duplicates.
    
    Args:
        expression (str): The input expression string
        
    Returns:
        list: Sorted list of unique keywords found in the expression
    """
    # Remove newlines and extra whitespace
    expression = ' '.join(expression.split())
    
    # Remove parentheses and commas
    expression = expression.replace('(', ' ').replace(')', ' ').replace(',', ' ')
    
    # Split into words, strip whitespace, and filter out empty strings more strictly
    words = [word.strip() for word in expression.split()]
    words = [word for word in words if word and not word.isspace()]
        
    # Get unique words and sort them
    unique_words = sorted(set(words))
    
    return unique_words

# Example usage
if __name__ == "__main__":
    # Test expression
    test_expr = """
    rot90(apply(fork(combine, rbind(sfilter, matcher(identity, mostcolor(I))), 
    rbind(sfilter, compose(flip, matcher(identity, mostcolor(I))))), rot270(I)))
    """
    
    result = parse_expression(test_expr)
    print("Found keywords:")
    for keyword in result:
        print(f"- {keyword}")