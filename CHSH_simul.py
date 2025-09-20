import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# Function to compute CHSH value given measurement angles for two parties
def compute_chsh(theta_A1, theta_A2, theta_B1, theta_B2):
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    # Generate measurement operators for given angles
    def measurement_operator(theta):
        return np.cos(theta) * Z + np.sin(theta) * X
    
    A1 = measurement_operator(theta_A1)
    A2 = measurement_operator(theta_A2)
    B1 = measurement_operator(theta_B1)
    B2 = measurement_operator(theta_B2)
    
    # Define singlet state |ψ> = (|01> - |10>) / sqrt(2)
    psi = np.array([0, 1, -1, 0]) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    
    # Helper for expectation <A ⊗ B>
    def expect(A, B):
        op = np.kron(A, B)
        return np.real(np.trace(rho @ op))
    
    # Compute CHSH combination
    S = expect(A1, B1) + expect(A1, B2) + expect(A2, B1) - expect(A2, B2)
    return S

# Generate random angle settings and compute CHSH values
num_samples = 10000
results = []
np.random.seed(42)
for _ in range(num_samples):
    angles = np.random.uniform(0, 2*np.pi, 4)
    S_val = compute_chsh(*angles)
    results.append({
        'theta_A1': angles[0],
        'theta_A2': angles[1],
        'theta_B1': angles[2],
        'theta_B2': angles[3],
        'CHSH': S_val,
        'Quantum Violation': S_val > 2
    })

df = pd.DataFrame(results)

print(df.head())

df.to_csv('chsh_simulation_results.csv', index=False)


# Plot the distribution of CHSH values
plt.figure(figsize=(10, 6))
sns.histplot(df['CHSH'], bins=30, kde=True, color='blue', label='CHSH Values')
plt.axvline(x=2, color='red', linestyle='--', label='Classical Limit (S=2)')
plt.title('Distribution of CHSH Values')
plt.xlabel('CHSH Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()