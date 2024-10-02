import itertools

def generate_binary_numbers(n):
    return [''.join(bits) for bits in itertools.product('01', repeat=n)]

n = 4  # n is the number of particles. you can change the value as needed.
spin_chains = generate_binary_numbers(n)
print(spin_chains)

def generate_zero_list(n):
    return [0] * n

h_list = generate_zero_list(n)
print(h_list)

def generate_zero_matrix(n):
    return [[0] * n for _ in range(n)]

J_list = generate_zero_matrix(n)
print(J_list)

# J_list[0][1] = ...
# J_list[1][2] = ...
# ... (setting all parameters J_ij)
# h_list[0] = ...
# h_list[1] = ...
# ... (setting all parameters h_i)

A = 1 # Annealing function. you can change the value as needed.
B = 1 # Annealing function. you can change the value as needed.

def generate_zero_matrix(n):
    return [[0] * m for _ in range(m)]

m = 2**n  # m is the number of configurations/basis vectors
hamiltonian_matrix = generate_zero_matrix(m)

def differ_by_one_bit(bin1, bin2):
    int1 = int(bin1, 2)
    int2 = int(bin2, 2)
    
    xor_result = int1 ^ int2
    
    if bin(xor_result).count('1') == 1:
        differing_bit_position = n - xor_result.bit_length() 
        return True, differing_bit_position
    else:
        return False, None

for i in range(0,m):
    for j in range(0,m):
        bin1 = spin_chains[i]
        bin2 = spin_chains[j]
        result, position = differ_by_one_bit(bin1, bin2)
        if result == True:
            hamiltonian_matrix[i][j] = hamiltonian_matrix[i][j] - A/2

for i in range(0,m):
    for j in range(0,n):
		    hamiltonian_matrix[i][i] = hamiltonian_matrix[i][i] + (B/2) * h_list[j] * (-1)**(int(spin_chains[i][j])) 

for i in range(0,m):
    for j in range(0,n-1):
		    for k in range(j+1,n):
				    hamiltonian_matrix[i][i] = hamiltonian_matrix[i][i] + (B/2) * J_list[j][k] * ((-1)**(int(spin_chains[i][j])*int(spin_chains[i][k])))

print(hamiltonian_matrix)









