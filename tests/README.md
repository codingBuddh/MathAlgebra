# MathAlgebra Test Suite

This directory contains comprehensive test suites for the MathAlgebra library, with a particular focus on matrix multiplication performance analysis and visualization.

## Test Files

- `test_matrix_multiplication.py`: Comprehensive test suite for matrix multiplication algorithms

## Performance Analysis

The test suite includes extensive performance analysis across different matrix types and sizes:

### Matrix Types Tested
- Dense Matrices
- Sparse Matrices (90% and 95% sparsity)
- Band Matrices (bandwidth 5 and 10)
- Toeplitz Matrices
- Circulant Matrices
- Hankel Matrices
- Symmetric Matrices
- Persymmetric Matrices
- Orthogonal Matrices
- Upper/Lower Triangular Matrices

### Algorithms Tested
- Naive Multiplication (O(n³))
- Strassen's Algorithm (O(n^2.807))
- Block Matrix Multiplication
- Sparse Matrix Multiplication
- Smart Multiplication (Automatic algorithm selection)

## Visualizations

All test results are saved in the `tests/results` directory. Here are the key visualizations generated:

### Performance Analysis
- [`performance_scaling.png`](results/performance_scaling.png): Performance scaling across different matrix sizes
- [`performance_comparison.png`](results/performance_comparison.png): Direct comparison of algorithm performance
- [`algorithm_transitions.png`](results/algorithm_transitions.png): Analysis of performance at algorithm transition points

### Memory Usage
- [`memory_usage.png`](results/memory_usage.png): Memory consumption analysis
- [`memory_analysis.png`](results/memory_analysis.png): Detailed memory usage patterns
- [`memory_by_size.png`](results/memory_by_size.png): Memory scaling with matrix size

### Efficiency Metrics
- [`efficiency.png`](results/efficiency.png): Computational efficiency analysis
- [`efficiency_by_size.png`](results/efficiency_by_size.png): Efficiency scaling with matrix size
- [`efficiency_heatmap.png`](results/efficiency_heatmap.png): Heatmap of algorithm efficiency

### Time Analysis
- [`time_analysis.png`](results/time_analysis.png): Execution time analysis
- [`time_analysis_detailed.png`](results/time_analysis_detailed.png): Detailed timing breakdown

## Results and Reports

The test suite generates detailed reports in both CSV and text formats:

- [`performance_results.csv`](results/performance_results.csv): Raw performance data
- [`performance_summary.txt`](results/performance_summary.txt): Summary of performance metrics
- [`time_analysis_results.csv`](results/time_analysis_results.csv): Detailed timing data
- [`time_analysis_summary.txt`](results/time_analysis_summary.txt): Summary of timing analysis

## Key Findings

1. **Algorithm Selection**:
   - Naive multiplication is fastest for matrices < 64×64
   - Strassen's algorithm performs best for 64×64 to 512×512
   - Block multiplication excels for matrices > 512×512
   - Sparse multiplication is optimal when sparsity > 80%

2. **Memory Efficiency**:
   - Block multiplication shows the best memory efficiency
   - Strassen's algorithm has higher memory overhead
   - Sparse multiplication uses minimal memory for sparse matrices

3. **Performance Boundaries**:
   - Clear performance transitions at 64×64 and 512×512
   - Algorithm selection significantly impacts performance
   - Memory usage scales differently for each algorithm

## Running Tests

To run the complete test suite:

```bash
python -m pytest tests/test_matrix_multiplication.py -v
```

For specific test cases:

```bash
# Run only performance analysis
python -m pytest tests/test_matrix_multiplication.py::TestMatrixMultiplication::test_comprehensive_performance_analysis -v

# Run time-based analysis
python -m pytest tests/test_matrix_multiplication.py::TestMatrixMultiplication::test_time_based_analysis -v
```

## Dependencies

All required dependencies are listed in `requirements.txt`. Key dependencies include:
- numpy>=1.20.0
- scipy>=1.7.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- pytest>=6.0.0
- psutil>=5.8.0
- seaborn>=0.11.0

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Include performance measurements using the `@measure_performance` decorator
3. Add visualizations for new metrics
4. Update this README with any new visualizations or metrics

## Notes

- All visualizations are automatically generated during test execution
- Memory measurements include garbage collection to ensure accuracy
- Performance metrics are averaged over multiple runs
- System information is included in test reports for reproducibility 