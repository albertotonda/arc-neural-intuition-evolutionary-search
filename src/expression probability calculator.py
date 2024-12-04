import json
from typing import Dict, List, Union, Tuple, Set
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProbStep:
    """Stores information about each probability calculation step"""
    depth: int
    expr: str
    operation: str
    log_prob: float
    details: str
    children: List['ProbStep']

def parse_expression(expr: str) -> Tuple[str, List[str]]:
    """Parse a DSL expression into function name and arguments"""
    if '(' not in expr:  # Handle terminal nodes (no parentheses)
        return expr, []
    
    fname = expr[:expr.index('(')]
    args_str = expr[expr.index('(')+1:expr.rindex(')')]
    
    args = []
    paren_count = 0
    current_arg = ''
    
    for char in args_str:
        if char == '(' and paren_count >= 0:
            paren_count += 1
        elif char == ')' and paren_count >= 0:
            paren_count -= 1
            
        if char == ',' and paren_count == 0:
            args.append(current_arg.strip())
            current_arg = ''
        else:
            current_arg += char
            
    if current_arg:
        args.append(current_arg.strip())
        
    return fname, args

def calculate_expression_log_probability(expr: str, node_tree: Dict, forbidden: Set[str]) -> Tuple[float, ProbStep]:
    """Calculate the log10 probability of generating a specific expression with forbidden keywords"""
    
    def filter_valid_lists(children_lists: List[List[str]], forbidden: Set[str]) -> List[List[str]]:
        """Filter out children lists containing forbidden keywords"""
        return [children for children in children_lists 
                if not any(child in forbidden for child in children)]
    
    def calc_node_log_probability(expr: str, depth: int = 0) -> Tuple[float, ProbStep]:
        # Parse the expression
        fname, args = parse_expression(expr)
        
        # Check if current function is forbidden
        if fname in forbidden:
            return float('-inf'), ProbStep(
                depth=depth,
                expr=expr,
                operation="Forbidden node",
                log_prob=float('-inf'),
                details=f"Function '{fname}' is in forbidden list",
                children=[]
            )
        
        # If no parentheses, treat as terminal node regardless of whether it appears 
        # in the tree or not
        if not args:
            step = ProbStep(
                depth=depth,
                expr=expr,
                operation="Terminal node",
                log_prob=0.0,  # log10(1) = 0
                details=f"Terminal node '{fname}' - probability 1.0 (log10_prob = 0.0)",
                children=[]
            )
            return 0.0, step
            
        # At this point we have a function call (has parentheses), so check the tree
        if fname not in node_tree:
            step = ProbStep(
                depth=depth,
                expr=expr,
                operation="Invalid function call",
                log_prob=float('-inf'),
                details=f"Function '{fname}' not found in node tree",
                children=[]
            )
            return float('-inf'), step
            
        # Get all possible children combinations for this node
        all_children_lists = node_tree[fname]
            
        # Filter out children lists containing forbidden keywords
        valid_children_lists = filter_valid_lists(all_children_lists, forbidden)
        
        if not valid_children_lists:
            step = ProbStep(
                depth=depth,
                expr=expr,
                operation="Invalid due to restrictions",
                log_prob=float('-inf'),
                details=f"No valid children lists for '{fname}' after forbidden filtering",
                children=[]
            )
            return float('-inf'), step
            
        # Find exact matching children lists among valid ones and count their occurrences
        matching_lists = []
        for child_list in valid_children_lists:
            if len(child_list) == len(args):
                # Check if all actual arguments are in the possible children for this position
                is_valid = True
                for i, arg_value in enumerate(args):
                    arg_name, _ = parse_expression(arg_value)
                    if arg_name != child_list[i]:
                        is_valid = False
                        break
                if is_valid:
                    matching_lists.append(child_list)
        
        if not matching_lists:
            step = ProbStep(
                depth=depth,
                expr=expr,
                operation="Invalid expression",
                log_prob=float('-inf'),
                details=f"No valid children lists match '{fname}({', '.join(args)})'",
                children=[]
            )
            return float('-inf'), step
            
        # Count occurrences of each unique children list
        unique_valid_lists = {}  # dict to store counts of unique lists
        for lst in valid_children_lists:
            lst_tuple = tuple(lst)  # convert list to tuple for hashing
            unique_valid_lists[lst_tuple] = unique_valid_lists.get(lst_tuple, 0) + 1
            
        matching_tuples = [tuple(lst) for lst in matching_lists]
        matching_count = sum(unique_valid_lists[t] for t in set(matching_tuples))
        total_count = sum(unique_valid_lists.values())
            
        # Calculate log10 probability for this node considering duplicates
        log_prob_this_node = math.log10(matching_count) - math.log10(total_count)
        
        # Calculate log10 probability of children
        child_steps = []
        log_prob_children = 0.0
        for arg in args:
            child_log_prob, child_step = calc_node_log_probability(arg, depth + 1)
            if child_log_prob == float('-inf'):
                return float('-inf'), child_step
            log_prob_children += child_log_prob
            child_steps.append(child_step)
            
        total_log_prob = log_prob_this_node + log_prob_children
        
        step = ProbStep(
            depth=depth,
            expr=expr,
            operation=f"Node '{fname}'",
            log_prob=total_log_prob,
            details=(f"Valid children lists occurrences: {total_count}\n" +
                    f"{'  ' * depth}Matching children lists occurrences: {matching_count}\n" +
                    f"{'  ' * depth}Node choice: log10({matching_count}/{total_count}) = {log_prob_this_node:.4f}\n" +
                    f"{'  ' * depth}Children log10 prob: {log_prob_children:.4f}\n" +
                    f"{'  ' * depth}Total log10 prob: {total_log_prob:.4f}"),
            children=child_steps
        )
        
        return total_log_prob, step

    # Calculate root probability
    root_lists = node_tree.get('root', [[]])
    valid_root_lists = filter_valid_lists(root_lists, forbidden)
    
    if not valid_root_lists:
        return float('-inf'), ProbStep(
            depth=0,
            expr=expr,
            operation="Invalid root",
            log_prob=float('-inf'),
            details="No valid root functions after forbidden filtering",
            children=[]
        )
    
    # Count occurrences of each valid root function
    root_counts = {}
    for root_list in valid_root_lists:
        for root_func in root_list:
            root_counts[root_func] = root_counts.get(root_func, 0) + 1
    
    fname, _ = parse_expression(expr)
    if fname not in root_counts:
        return float('-inf'), ProbStep(
            depth=0,
            expr=expr,
            operation="Invalid root",
            log_prob=float('-inf'),
            details=f"Expression doesn't start with valid root function",
            children=[]
        )
    
    # Calculate root probability considering duplicates
    total_root_occurrences = sum(root_counts.values())
    root_log_prob = math.log10(root_counts[fname]) - math.log10(total_root_occurrences)
    
    node_log_prob, node_step = calc_node_log_probability(expr)
    
    if node_log_prob == float('-inf'):
        return float('-inf'), node_step
        
    total_log_prob = root_log_prob + node_log_prob
    
    root_step = ProbStep(
        depth=0,
        expr=expr,
        operation="Root selection",
        log_prob=total_log_prob,
        details=(f"Total root occurrences: {total_root_occurrences}\n" +
                f"This root '{fname}' occurrences: {root_counts[fname]}\n" +
                f"Root choice: log10({root_counts[fname]}/{total_root_occurrences}) = {root_log_prob:.4f}\n" +
                f"Expression log10 prob: {node_log_prob:.4f}\n" +
                f"Total log10 prob: {total_log_prob:.4f}"),
        children=[node_step]
    )
    
    return total_log_prob, root_step

def print_probability_tree(step: ProbStep, indent: str = ""):
    """Print the probability calculation tree in a readable format"""
    print(f"{indent}├─ {step.operation}")
    print(f"{indent}│  Expression: {step.expr}")
    for detail_line in step.details.split('\n'):
        print(f"{indent}│  {detail_line}")
    
    for child in step.children:
        print_probability_tree(child, indent + "│  ")

def extract_keywords(expr: str) -> Set[str]:
    """Extract all unique keywords from an expression"""
    keywords = set()
    
    def process_expr(expr: str):
        fname, args = parse_expression(expr)
        keywords.add(fname)
        for arg in args:
            process_expr(arg)
    
    process_expr(expr)
    return keywords

def main():
    # Load node tree
    with open('node_tree.json', 'r') as f:
        node_tree = json.load(f)
    
    # Get all possible keywords from node tree
    all_keywords = set()
    for func_name, children_lists in node_tree.items():
        if func_name != 'root':
            all_keywords.add(func_name)
            for children in children_lists:
                all_keywords.update(children)
    
    # Get forbidden keywords
    print("Enter forbidden keywords (comma-separated) or press Enter to skip:")
    forbidden_input = input("> ")
    forbidden = {keyword.strip() for keyword in forbidden_input.split(',')} if forbidden_input else set()
    
    # Get expression from user
    expr = input("\nEnter a DSL expression: ")
    
    try:
        # First calculate probability with user-specified forbidden keywords
        print("\nProbability calculation with user-specified restrictions:")
        print("=====================================================")
        log_probability, prob_tree = calculate_expression_log_probability(expr, node_tree, forbidden)
        print_probability_tree(prob_tree)
        
        print("\nSummary with user-specified restrictions:")
        print("----------------------------------------")
        print(f"Log10 probability: {log_probability:.4f}")
        if log_probability != float('-inf'):
            print(f"Probability: {10**log_probability:.2e}")
            print(f"Approximately 1 in {int(1/(10**log_probability)):,} generations")
        else:
            print("Expression impossible with given restrictions")
            
        # Now calculate probability with self-restriction
        expr_keywords = extract_keywords(expr)
        self_forbidden = all_keywords - expr_keywords
        
        print("\nProbability calculation with self-restriction:")
        print("(only keywords in the expression are allowed)")
        print("=============================================")
        log_probability_self, prob_tree_self = calculate_expression_log_probability(expr, node_tree, self_forbidden)
        print_probability_tree(prob_tree_self)
        
        print("\nSummary with self-restriction:")
        print("------------------------------")
        print(f"Allowed keywords ({len(expr_keywords)}): {sorted(expr_keywords)}")
        print(f"Log10 probability: {log_probability_self:.4f}")
        if log_probability_self != float('-inf'):
            print(f"Probability: {10**log_probability_self:.2e}")
            print(f"Approximately 1 in {int(1/(10**log_probability_self)):,} generations")
        else:
            print("Expression impossible with self-restriction")
            
    except Exception as e:
        print(f"Error calculating probability: {str(e)}")
        print("Make sure the expression is valid DSL syntax")

if __name__ == "__main__":
    main()    