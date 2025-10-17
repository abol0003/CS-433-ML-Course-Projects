# Test script to verify CSV output format
import numpy as np

# Simulate some results
mock_results = [
    {
        'lambda': 0.001,
        'learning_rate': 0.1,
        'max_iters': 1000,
        'mean_accuracy': 0.823456,
        'std_accuracy': 0.012345,
        'mean_precision': 0.812345,
        'std_precision': 0.014567,
        'mean_recall': 0.819012,
        'std_recall': 0.012678,
        'mean_f1': 0.815678,
        'std_f1': 0.013456,
        'optimal_threshold': 0.523456
    },
    {
        'lambda': 0.001,
        'learning_rate': 0.01,
        'max_iters': 1000,
        'mean_accuracy': 0.834567,
        'std_accuracy': 0.011234,
        'mean_precision': 0.823456,
        'std_precision': 0.013456,
        'mean_recall': 0.830123,
        'std_recall': 0.011567,
        'mean_f1': 0.826789,
        'std_f1': 0.012345,
        'optimal_threshold': 0.534567
    }
]

# Write CSV in the exact format
csv_path = 'test_output.csv'
with open(csv_path, 'w') as f:
    # Write header with exact column order
    f.write('lambda,gamma,max_iters,acc_mean,acc_std,prec_mean,prec_std,rec_mean,rec_std,f1_mean,f1_std,optimal_threshold\n')
    
    # Write data rows
    for r in mock_results:
        f.write(f"{r['lambda']:.6f},{r['learning_rate']:.6f},{r['max_iters']},"
               f"{r['mean_accuracy']:.6f},{r['std_accuracy']:.6f},"
               f"{r['mean_precision']:.6f},{r['std_precision']:.6f},"
               f"{r['mean_recall']:.6f},{r['std_recall']:.6f},"
               f"{r['mean_f1']:.6f},{r['std_f1']:.6f},"
               f"{r['optimal_threshold']:.6f}\n")

print(f"✅ Created test CSV: {csv_path}")
print("\nCSV Preview:")
with open(csv_path, 'r') as f:
    print(f.read())
