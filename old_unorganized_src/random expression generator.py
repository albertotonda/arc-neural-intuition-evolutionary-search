import json
import random
from typing import List, Dict, Set

def generate_random_expression(node_tree: Dict, max_depth: int = 10, forbidden: Set[str] = set()) -> str:
    """
    Generate random DSL expressions while avoiding specified keywords/functions
    
    Args:
        node_tree: The DSL node tree structure
        max_depth: Maximum depth of generated expression
        forbidden: Set of keywords/functions to avoid
        
    Returns:
        A randomly generated expression string
    """
    def expand_node(node: str, current_depth: int = 0) -> str:
        # Stop if max depth reached or node not in tree
        if node not in node_tree:
            return node
        
        if current_depth >= max_depth:
            return node
        
        # Select a random child list for this node
        children_lists = node_tree[node]
        if not children_lists:
            return node
        
        # Filter out children lists containing forbidden nodes
        valid_children_lists = []
        for children in children_lists:
            if not any(child in forbidden for child in children):
                valid_children_lists.append(children)
                
        # If no valid children lists, return node as is
        if not valid_children_lists:
            return node
        
        # Choose a random valid children list
        children = random.choice(valid_children_lists)
        
        # Recursively expand children
        expanded_children = [expand_node(child, current_depth + 1).lstrip('_') for child in children]
        
        # Construct function call
        return f"{node}({', '.join(expanded_children)})"
    
    # Start from root node
    root_lists = node_tree.get('root', [[]])
    
    # Filter out root lists containing forbidden nodes
    valid_root_lists = []
    for root_list in root_lists:
        if not any(node in forbidden for node in root_list):
            valid_root_lists.append(root_list)
    
    if not valid_root_lists:
        raise ValueError("No valid expressions possible with given forbidden keywords")
    
    # Choose random valid root
    root_list = random.choice(valid_root_lists)
    root = random.choice(root_list)
    
    return expand_node(root)

def load_node_tree(filename: str) -> Dict:
    """Load node tree from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def print_available_keywords(node_tree: Dict):
    """Print all available keywords in the DSL"""
    keywords = set()
    # Add all nodes
    keywords.update(node_tree.keys())
    # Add all children
    for children_lists in node_tree.values():
        for children in children_lists:
            keywords.update(children)
    
    print("\nAvailable keywords:")
    print("------------------")
    for keyword in sorted(keywords):
        print(keyword)

def main():
    # Load node tree
    try:
        node_tree = load_node_tree('node_tree.json')
        #print('node tree len:', len(node_tree))
        print(node_tree.keys())
    except FileNotFoundError:
        print("Error: node_tree.json not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON in node_tree.json")
        return

    # Print available keywords
    print_available_keywords(node_tree)
    
    # Get forbidden keywords from user
    print("\nEnter forbidden keywords (comma-separated) or press Enter to skip:")
    forbidden_input = input("> ")
    forbidden = {keyword.strip() for keyword in forbidden_input.split(',')} if forbidden_input else set()
    
    # Get number of expressions to generate
    try:
        num_expressions = int(input("\nNumber of expressions to generate: "))
    except ValueError:
        print("Invalid number, defaulting to 5")
        num_expressions = 5
    
    # Get maximum depth
    try:
        max_depth = int(input("Maximum expression depth (default 10): "))
    except ValueError:
        print("Invalid depth, defaulting to 10")
        max_depth = 10
    
    print("\nGenerating expressions...")
    print("------------------------")
    
    try:
        for i in range(num_expressions):
            expr = generate_random_expression(node_tree, max_depth, forbidden)
            print(f"{i+1}. {expr}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()