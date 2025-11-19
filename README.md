# ============================================================
# MATHEMATICS FOR COMPUTING – FULL PRACTICAL (Q1–Q13)
# Each question statement is included as comments before its code.
# Save as: mathematics_for_computing_practical.py
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# Q1. Create and transform vectors and matrices (the transpose vector (matrix) conjugate)
# a. transpose of a vector (matrix)
# - Create horizontal and vertical vectors
# - Create a complex matrix, show transpose and conjugate transpose (Hermitian)
# ============================================================
print("\n================= Q1. Create & Transform Vectors/Matrix =================")
h = np.array([1+2j, 2-1j, 3.0])
print("Horizontal vector h:", h)
v = h.reshape(-1, 1)
print("Vertical vector v (column):\n", v)

A = np.array([
    [1, 2+1j, 3],
    [4, 5, 6-2j],
    [7, 8, 9]
])
print("Matrix A:\n", A)
print("Transpose A^T:\n", A.T)
print("Conjugate transpose A^H:\n", np.conjugate(A.T))


# ============================================================
# Q2. Generate the matrix into echelon form and find its rank.
# - Use sympy to compute RREF (reduced row echelon form) and rank
# ============================================================
print("\n================= Q2. Echelon Form & Rank =================")
M = sp.Matrix([[1, 2, 0],
               [2, 4, 1],
               [3, 6, 3]])
print("Matrix M:\n", M)
rrefM, pivots = M.rref()
print("RREF:\n", rrefM)
print("Pivot columns:", pivots)
print("Rank:", M.rank())


# ============================================================
# Q3. Find cofactors, determinant, adjoint and inverse of a matrix.
# - Compute determinant, cofactor matrix, classical adjoint and inverse (if exists)
# ============================================================
print("\n================= Q3. Cofactor, Determinant, Adjoint, Inverse =================")
B = sp.Matrix([[2, 3, 1],
               [1, 0, 4],
               [5, 2, 2]])
print("Matrix B:\n", B)
detB = B.det()
print("det(B) =", detB)
cof = B.cofactor_matrix()
print("Cofactor matrix:\n", cof)
adj = cof.T
print("Adjoint (classical adjugate):\n", adj)
if detB != 0:
    print("Inverse B^-1:\n", B.inv())
else:
    print("B is singular, inverse does not exist.")


# ============================================================
# Q4. Solve a system of Homogeneous and non-homogeneous equations using Gauss elimination method.
# - Solve Ax = b (non-homogeneous) and find nullspace for Ax = 0 (homogeneous)
# ============================================================
print("\n================= Q4. Solve Homogeneous & Non-Homogeneous System =================")
A1 = sp.Matrix([[2, -1, 1],
                [3, 2, -4],
                [1, 1, 1]])
b1 = sp.Matrix([1, -2, 6])
print("Solve Ax = b for A =\n", A1, "\nand b =", b1)
sol = A1.LUsolve(b1)
print("Solution x:", sol)

A2 = sp.Matrix([[1, 2, 3],
                [2, 4, 6],
                [1, 1, 1]])
print("Homogeneous system A2 x = 0, A2 =\n", A2)
print("Nullspace basis for A2 (solutions to Ax=0):", A2.nullspace())


# ============================================================
# Q5. Solve a system of Homogeneous equations using the Gauss Jordan method.
# - Use rref to obtain solutions of C x = 0
# ============================================================
print("\n================= Q5. Gauss-Jordan Homogeneous System =================")
C = sp.Matrix([[1, 2, -1, 0],
               [2, 4, -2, 0],
               [3, 6, -3, 0]])
print("Matrix C:\n", C)
rrefC, pivotsC = C.rref()
print("RREF of C:\n", rrefC)
print("Pivots:", pivotsC)
print("Nullspace basis (solutions of Cx=0):", C.nullspace())


# ============================================================
# Q6. Generate basis of column space, null space, row space and left null space of a matrix space.
# - Compute columnspace, rowspace, nullspace and left nullspace (nullspace of transpose)
# ============================================================
print("\n================= Q6. Column, Row, Null & Left Null Space =================")
D = sp.Matrix([[1, 2, 3],
               [2, 4, 6],
               [1, 0, 1]])
print("Matrix D:\n", D)
print("Column space basis:", D.columnspace())
print("Row space basis:", D.rowspace())
print("Null space basis:", D.nullspace())
print("Left null space basis (nullspace of D^T):", D.T.nullspace())


# ============================================================
# Q7. Check the linear dependence of vectors. Generate a linear combination and find the transition matrix.
# - Given vectors in R^n, form matrix with columns, compute rank to check dependence
# - Show a sample linear combination and a change-of-basis (transition) matrix
# ============================================================
print("\n================= Q7. Linear Dependence & Transition Matrix =================")
v1 = sp.Matrix([1, 2, 3])
v2 = sp.Matrix([2, 4, 6])
v3 = sp.Matrix([0, 1, 1])
M_cols = sp.Matrix.hstack(v1, v2, v3)
print("Matrix with vectors as columns:\n", M_cols)
print("Rank:", M_cols.rank(), "Number of vectors:", M_cols.cols)
if M_cols.rank() < M_cols.cols:
    print("=> Vectors are linearly dependent")
else:
    print("=> Vectors are linearly independent")

c1, c2, c3 = sp.symbols('c1 c2 c3')
lin_comb = c1*v1 + c2*v2 + c3*v3
print("General linear combination c1*v1 + c2*v2 + c3*v3 =\n", lin_comb)

# Example: choose independent vectors as a basis (v1, v3, e1)
e1 = sp.Matrix([1, 0, 0])
B = sp.Matrix.hstack(v1, v3, e1)
print("Basis matrix B (columns form basis):\n", B)
print("Transition matrix P (B -> standard basis):\n", B)
print("Inverse P_inv (standard -> B):\n", B.inv())


# ============================================================
# Q8. Find the orthonormal basis of a given vector space using the Gram-Schmidt orthogonalization process.
# - Use QR decomposition to obtain orthogonal (orthonormal) basis
# ============================================================
print("\n================= Q8. Orthonormal Basis (Gram-Schmidt / QR) =================")
vectors = [sp.Matrix([1, 1, 0]), sp.Matrix([1, 0, 1]), sp.Matrix([0, 1, 1])]
M3 = sp.Matrix.hstack(*vectors)
print("Original vectors as columns:\n", M3)
Q, R = M3.QRdecomposition()
print("Orthogonal Q matrix:\n", Q)
print("Upper-triangular R matrix:\n", R)
print("Orthonormal basis (columns of Q) and their norms:")
for i in range(Q.shape[1]):
    vec = Q.col(i)
    print(vec, "norm =", sp.sqrt(vec.dot(vec)))


# ============================================================
# Q9. Check the diagonalizable property of matrices and find eigenvalues; verify Cayley-Hamilton theorem.
# - Find eigenvalues/eigenvectors, check diagonalizability, and verify characteristic polynomial p(E)=0
# ============================================================
print("\n================= Q9. Diagonalizability & Cayley-Hamilton =================")
E = sp.Matrix([[5, 4, 2],
               [0, 1, 0],
               [0, 0, 3]])
print("Matrix E:\n", E)
print("Eigenvalues & eigenvectors:", E.eigenvects())
print("Is diagonalizable?", E.is_diagonalizable())
if E.is_diagonalizable():
    P_diag, D_diag = E.diagonalize()
    print("P (eigenvectors):\n", P_diag)
    print("D (diagonal):\n", D_diag)

x = sp.symbols('x')
charpoly = E.charpoly(x).as_expr()
print("Characteristic polynomial p(x):", charpoly)
pE = sp.expand(charpoly.subs(x, E))
print("p(E) (should be zero matrix):\n", pE)


# ============================================================
# Q10. Application: Coding and decoding messages using nonsingular matrices.
# - Code and decode message "Linear Algebra is fun" using a 2x2 key matrix (mod 26 arithmetic)
# ============================================================
print("\n================= Q10. Coding & Decoding using Non-Singular Matrix =================")
def text_to_nums(s):
    s = ''.join(ch for ch in s.upper() if ch.isalpha())
    return [ord(ch)-65 for ch in s]

def nums_to_text(nums):
    return ''.join(chr((n % 26)+65) for n in nums)

msg = "Linear Algebra is fun"
nums = text_to_nums(msg)
if len(nums) % 2:
    nums.append(23)   # pad with 'X' -> 23

K = np.array([[3, 3],
              [2, 5]])
print("Key K:\n", K)

encoded = []
for i in range(0, len(nums), 2):
    block = np.array(nums[i:i+2])
    cipher_block = (K.dot(block) % 26)
    encoded.extend(cipher_block.tolist())

ciphertext = nums_to_text(encoded)
print("Ciphertext:", ciphertext)

# decode
def egcd(a,b):
    if b==0: return (1,0,a)
    x,y,g = egcd(b, a%b)
    return (y, x - (a//b)*y, g)

det = int(round(np.linalg.det(K)))
_, inv_det_candidate, g = egcd(det, 26)
if g == 1:
    inv_det = inv_det_candidate % 26
    adj = np.array([[K[1,1], -K[0,1]], [-K[1,0], K[0,0]]])
    K_inv_mod26 = (inv_det * adj) % 26

    decoded = []
    for i in range(0, len(encoded), 2):
        block = np.array(encoded[i:i+2])
        plain_block = (K_inv_mod26.dot(block) % 26)
        decoded.extend(plain_block.tolist())
    decoded_text = nums_to_text(decoded)
    print("Decoded text:", decoded_text)
else:
    print("Key matrix K not invertible modulo 26, can't decode.")


# ============================================================
# Q11. Compute Gradient of a scalar field.
# - Given f(x,y,z) compute ∇f
# ============================================================
print("\n================= Q11. Gradient of Scalar Field =================")
x, y, z = sp.symbols('x y z', real=True)
f = x**2 * y + sp.sin(y*z) + sp.exp(x*z)
grad_f = [sp.diff(f, var) for var in (x, y, z)]
print("Scalar field f(x,y,z):", f)
print("Gradient ∇f =", sp.Matrix(grad_f))


# ============================================================
# Q12. Compute Divergence of a vector field.
# - Given vector field F = (P, Q, R), compute ∇·F
# ============================================================
print("\n================= Q12. Divergence of Vector Field =================")
P_expr = x*y**2
Q_expr = sp.sin(x*z)
R_expr = sp.exp(y*z)
divF = sp.diff(P_expr, x) + sp.diff(Q_expr, y) + sp.diff(R_expr, z)
print("Vector field F = (P,Q,R):", (P_expr, Q_expr, R_expr))
print("Divergence ∇·F =", sp.simplify(divF))


# ============================================================
# Q13. Compute Curl of a vector field.
# - Given F = (P, Q, R), compute ∇×F
# ============================================================
print("\n================= Q13. Curl of Vector Field =================")
P_c = y*z
Q_c = x*z
R_c = x*y
curl = sp.Matrix([sp.diff(R_c, y) - sp.diff(Q_c, z),
                  sp.diff(P_c, z) - sp.diff(R_c, x),
                  sp.diff(Q_c, x) - sp.diff(P_c, y)])
print("Vector field F:", (P_c, Q_c, R_c))
print("Curl ∇×F =", curl)

print("\n================= END OF PRACTICAL =================")
