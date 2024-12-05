import random
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import colorama
from colorama import Fore, Back, Style
import dsl

@dataclass
class TestResult:
    success: bool
    input_grid: Tuple[Tuple[int, ...], ...]
    output_grid: Optional[Tuple[Tuple[int, ...], ...]]
    solver: str
    error: Optional[str] = None

class DSLGenerator:
    def __init__(self):
        # Grid -> Grid operations
        self.grid_ops = [
            # Basic grid transformations
            (lambda g: f"rot90({g})", 2),
            (lambda g: f"rot180({g})", 2),
            (lambda g: f"rot270({g})", 2),
            (lambda g: f"hmirror({g})", 2),
            (lambda g: f"vmirror({g})", 2),
            (lambda g: f"trim({g})", 1),
            
            # Grid splits and concatenations
            (lambda g: f"vconcat(tophalf({g}), bottomhalf({g}))", 1),
            (lambda g: f"hconcat(lefthalf({g}), righthalf({g}))", 1),
            
            # Scaling operations
            (lambda g: f"upscale({g}, TWO)", 1),
            (lambda g: f"downscale(upscale({g}, TWO), TWO)", 1),
            
            # Color operations with safe color values
            (lambda g: f"replace({g}, {self.random_int()}, {self.random_int()})", 2),
            (lambda g: f"switch({g}, {self.random_int()}, {self.random_int()})", 2),
            
            # Complex transformations using composition
            (lambda g: f"fill({g}, {self.random_int()}, {self.safe_indices_expr()})", 2),
            (lambda g: f"paint({g}, {self.safe_object_expr()})", 2),
            (lambda g: f"underfill({g}, {self.random_int()}, {self.safe_indices_expr()})", 1),
            (lambda g: f"underpaint({g}, {self.safe_object_expr()})", 1),
            
            # Color manipulation using arithmetic
            (lambda g: f"paint({g}, recolor({self.color_arithmetic()}, asobject({g})))", 1),
            
            # Conditional transformations
            (lambda g: self.generate_conditional_transform(g), 1)
        ]

    def random_int(self) -> str:
        return random.choice(['ZERO', 'ONE', 'TWO', 'THREE'])

    def safe_indices_expr(self) -> str:
        """Generate indices expressions that are guaranteed to be valid"""
        return random.choice([
            "ofcolor(I, ZERO)",
            "asindices(I)",
            "product(interval(ZERO, height(I), ONE), interval(ZERO, width(I), ONE))",
            "corners(asindices(I))",
            "connect(ulcorner(asindices(I)), lrcorner(asindices(I)))",
            "box(asindices(I))",
            "backdrop(asindices(I))",
            f"ofcolor(I, {self.random_int()})"
        ])

    def safe_object_expr(self) -> str:
        """Generate object expressions that are guaranteed to be valid"""
        return random.choice([
            "asobject(I)",
            f"recolor({self.random_int()}, asindices(I))",
            "normalize(asobject(I))",
            f"shift(asobject(I), {self.random_direction()})"
        ])

    def random_direction(self) -> str:
        return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    def color_arithmetic(self) -> str:
        """Generate safe color arithmetic expressions"""
        ops = [
            lambda x: f"increment({x})",
            lambda x: f"decrement({x})",
            lambda x: f"double({x})",
            lambda x: f"halve({x})",
            lambda x: f"add({x}, ONE)",
            lambda x: f"subtract({x}, ONE)",
            lambda x: f"multiply({x}, TWO)",
            lambda x: f"divide({x}, TWO)"
        ]
        return random.choice(ops)(self.random_int())

    def generate_conditional_transform(self, g: str) -> str:
        """Generate a conditional transformation using branch"""
        conditions = [
            f"square({g})",
            f"even(height({g}))",
            f"even(width({g}))",
            f"positive(size(ofcolor({g}, {self.random_int()})))"
        ]
        
        transforms = [
            f"rot90({g})",
            f"hmirror({g})",
            f"vmirror({g})",
            g
        ]
        
        cond = random.choice(conditions)
        true_transform = random.choice(transforms)
        false_transform = random.choice(transforms)
        
        return f"branch({cond}, {true_transform}, {false_transform})"

    def generate_solver(self) -> str:
        expr = "I"
        ops = random.randint(2, 3)  # Keep chain length reasonable for stability
        
        weighted_ops = []
        for op, weight in self.grid_ops:
            weighted_ops.extend([op] * int(weight * 10))
            
        for _ in range(ops):
            op = random.choice(weighted_ops)
            expr = op(expr)
        return expr

class GridVisualizer:
    def __init__(self):
        colorama.init()
        self.colors = [
            Back.BLACK,
            Back.RED,
            Back.GREEN,
            Back.YELLOW,
            Back.BLUE,
            Back.MAGENTA,
            Back.CYAN,
            Back.WHITE
        ]

    def visualize_grid(self, grid: Tuple[Tuple[int, ...], ...]) -> str:
        output = []
        for row in grid:
            row_str = []
            for val in row:
                color = self.colors[val % len(self.colors)]
                row_str.append(f"{color}{Fore.WHITE} {val} {Style.RESET_ALL}")
            output.append("".join(row_str))
        return "\n".join(output)

class DSLTester:
    def __init__(self):
        self.generator = DSLGenerator()
        self.visualizer = GridVisualizer()
        self.namespace = self._create_namespace()

    def _create_namespace(self) -> Dict[str, Any]:
        namespace = {
            'T': True, 'F': False,
            'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4,
            'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)
        }
        
        for name in dir(dsl):
            attr = getattr(dsl, name)
            if callable(attr) or isinstance(attr, (tuple, int, bool)):
                namespace[name] = attr
                
        return namespace

    def create_test_grid(self, size: Tuple[int, int], colors: int = 4) -> Tuple[Tuple[int, ...]]:
        return tuple(tuple(random.randint(0, colors-1) for _ in range(size[1])) 
                    for _ in range(size[0]))

    def run_test(self, grid: Tuple[Tuple[int, ...], ...], solver: str) -> TestResult:
        try:
            test_namespace = self.namespace.copy()
            test_namespace['I'] = grid
            
            result = eval(solver, test_namespace)
            
            if isinstance(result, tuple) and all(isinstance(row, tuple) for row in result):
                return TestResult(True, grid, result, solver)
            return TestResult(False, grid, None, solver, "Invalid output type")
        except Exception as e:
            return TestResult(False, grid, None, solver, f"{type(e).__name__}: {str(e)}")

    def run_tests(self, num_tests: int = 5):
        grid_sizes = [(3,3), (4,4), (5,5)]
        results = []

        for i in range(num_tests):
            size = random.choice(grid_sizes)
            grid = self.create_test_grid(size)
            solver = self.generator.generate_solver()
            result = self.run_test(grid, solver)
            results.append(result)
            self._print_test_result(i + 1, result)

        self._print_summary(results)

    def _print_test_result(self, test_num: int, result: TestResult):
        print(f"\n{Fore.CYAN}Test {test_num}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Solver:{Style.RESET_ALL} {result.solver}")
        
        print(f"\n{Fore.GREEN}Input Grid:{Style.RESET_ALL}")
        print(self.visualizer.visualize_grid(result.input_grid))
        
        if result.success:
            print(f"\n{Fore.GREEN}Output Grid:{Style.RESET_ALL}")
            print(self.visualizer.visualize_grid(result.output_grid))
            print(f"\n{Fore.GREEN}Success!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}Error: {result.error}{Style.RESET_ALL}")

    def _print_summary(self, results: List[TestResult]):
        success_count = sum(1 for r in results if r.success)
        print(f"\n{Fore.CYAN}Test Summary{Style.RESET_ALL}")
        print(f"Total tests: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")

if __name__ == "__main__":
    tester = DSLTester()
    tester.run_tests(10)