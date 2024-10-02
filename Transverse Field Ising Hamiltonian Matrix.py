import itertools

def generate_binary_numbers(n):
    return [''.join(bits) for bits in itertools.product('01', repeat=n)]

n = 4  # n is the number of particles. you can change the value as needed.
spin_chains = generate_binary_numbers(n)
print(spin_chains)

J=1 # J is the interaction strength. you can change the value as needed.
h=1 # h is the magnetic field. you can change the value as needed.

def generate_zero_matrix(n):
    return [[0] * m for _ in range(m)]

m = 2**n  # m is the number of configurations/basis vectors
hamiltonian_matrix = generate_zero_matrix(m)

def differ_by_one_bit(bin1, bin2):
    int1 = int(bin1, 2)
    int2 = int(bin2, 2)
    
    xor_result = int1 ^ int2
    
    return bin(xor_result).count('1') == 1

for i in range(0,m):
    for j in range(0,m):
		    bin1 = spin_chains[i]  
		    bin2 = spin_chains[j]  
		    result = differ_by_one_bit(bin1, bin2)
		    if result == True:
				    hamiltonian_matrix[i][j] = hamiltonian_matrix[i][j] - h

for i in range(0,m):
    for j in range(0,n-1):
		    hamiltonian_matrix[i][i] = hamiltonian_matrix[i][i] - J * ((-1)**(int(spin_chains[i][j])*int(spin_chains[i][j+1])))
			
print(hamiltonian_matrix)







