# Define the original data points
p1 = ([0.151, 0.151, 0, 0, 0, 0, 0, 0, 0, 0])
p2 = ([0.075, 0, 0.151, 0.151, 0.031, 0, 0, 0, 0, 0])
p3 = ([0, 0, 0, 0, 0.031, 0.151, 0.075, 0.151, 0, 0])

# Define the neighbors for each point
neighbors = {
    "p1": [p2, p3],
    "p2": [p1, p3],
    "p3": [p1, p2]
}

# Function to generate synthetic sample
def generate_synthetic_sample_manual(x, y, lambda_val):
    return x + lambda_val * (y - x)

# Generate synthetic sample for p1 with a specific lambda
lambda_val = 0.6
synthetic_p1 = generate_synthetic_sample_manual(p1, p2, lambda_val)

# Generate synthetic sample for p2 with a specific lambda
lambda_val = 0.7
synthetic_p2 = generate_synthetic_sample_manual(p2, p3, lambda_val)

# Generate synthetic sample for p3 with a specific lambda
lambda_val = 0.5
synthetic_p3 = generate_synthetic_sample_manual(p3, p1, lambda_val)

# Combine the synthetic samples into a DataFrame
synthetic_samples_manual = ([synthetic_p1, synthetic_p2, synthetic_p3],columns=['bagus', 'pemerintah', 'setuju', 'kalau', 'tutup', 'saja', 'tiktokshop', 'bangkrut', 'goblok', 'main'])
synthetic_samples_manual.index = ['synthetic_p1', 'synthetic_p2', 'synthetic_p3']
synthetic_samples_manual
