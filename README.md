# mathematics-for-computing-practical
# Q1. Create and transform vectors and matrices (the transpose vector (matrix) conjugate
a. transpose of a vector (matrix)

Input (code)

    import numpy as np
    # horizontal vector from list
    h = np.array([1+2j, 2-1j, 3.0])   
    print("Horizontal vector h:", h)
    # vertical vector (column)
    v = h.reshape(-1, 1)
    print("Vertical vector v (column):\n", v)
    # matrix creation
    A = np.array([[1, 2+1j, 3],
                  [4, 5,    6-2j],
                  [7, 8,    9]])
    print("Matrix A:\n", A)
    # transpose
    At = A.T
    print("Transpose A^T:\n", At)
    # conjugate transpose (Hermitian)
    A_h = np.conjugate(A.T)
    print("Conjugate transpose (A^H):\n", A_h)              
# Q2. Generate the matrix into echelon form and find its rank.       
Input (code)

    import numpy as np
    import sympy as sp
    
    M = sp.Matrix([[1, 2, 0],
                   [2, 4, 1],
                   [3, 6, 3]])
    print("Matrix M:\n", M)
    
    # Reduced row echelon form (rref) and pivot columns
    rref_matrix, pivots = M.rref()
    print("\nRREF of M:\n", rref_matrix)
    print("Pivot columns:", pivots)
    
    # Rank
    rank = M.rank()
    print("Rank of M:", rank)
# Q3. Find cofactors, determinant, adjoint and inverse of a matrix.
Input (code)

    import sympy as sp
    
    B = sp.Matrix([[2, 3, 1],
                   [1, 0, 4],
                   [5, 2, 2]])
    print("Matrix B:\n", B)
    
    detB = B.det()
    print("det(B) =", detB)
    
    # Cofactor matrix: matrix of cofactors (classical adjoint is transpose of cofactor)
    cofactor_matrix = B.cofactor_matrix()
    print("Cofactor matrix:\n", cofactor_matrix)
    
    # Adjoint (classical adjoint) is cofactor_matrix.T
    adjoint = cofactor_matrix.T
    print("Adjoint (classical adjoint):\n", adjoint)
    
    # Inverse if determinant non-zero
    if detB != 0:
        invB = B.inv()
        print("Inverse B^-1:\n", invB)
    else:
        print("B is singular, inverse does not exist.")
# Q4. Solve a system of Homogeneous and non-homogeneous equations using Gauss
a. elimination method.

Input (code)

    import numpy as np
    import sympy as sp
    
    # Example non-homogeneous system Ax = b
    A = sp.Matrix([[2, -1, 1],
                   [3, 2, -4],
                   [1, 1, 1]])
    b = sp.Matrix([1, -2, 6])
    print("Solve Ax=b with A:\n", A, "\nb =", b)
    
    # Solve with sympy (exact)
    x = A.LUsolve(b)
    print("Solution x:", x)
    
    # Example homogeneous system Ax = 0
    A2 = sp.Matrix([[1, 2, 3],
                    [2, 4, 6],
                    [1, 1, 1]])
    ns = A2.nullspace()   # nullspace gives basis of solutions
    print("\nNullspace basis for A2 (Ax=0):", ns)
# Q5. Solve a system of Homogeneous equations using the Gauss Jordan method.
Input (code)

    import sympy as sp
    
    C = sp.Matrix([[1, 2, -1, 0],
                   [2, 4, -2, 0],
                   [3, 6, -3, 0]])
    print("Matrix C:\n", C)
    
    rrefC, pivots = C.rref()
    print("RREF of C:\n", rrefC)
    print("Pivots:", pivots)
    
    # Extract nullspace (solutions to C x = 0)
    null = C.nullspace()
    print("Nullspace basis (solutions of Cx=0):", null)
# Q6. Generate basis of column space, null space, row space and left null space of a matrix space.
Input (code)

    import sympy as sp
    
    D = sp.Matrix([[1, 2, 3],
                   [2, 4, 6],
                   [1, 0, 1]])
    print("Matrix D:\n", D)
    
    col_space = D.columnspace()
    row_space = D.rowspace()
    null_space = D.nullspace()
    left_null_space = D.T.nullspace()   # left nullspace of D equals nullspace of D^T
    
    print("Column space basis:", col_space)
    print("Row space basis:", row_space)
    print("Null space basis:", null_space)
    print("Left null space basis:", left_null_space)
# Q7. Check the linear dependence of vectors. Generate a linear combination of given vectors of R^n/ matrices of the same size and find the transition matrix of given matrix space.
Input (code)

    import sympy as sp
    
    # Vectors in R3
    v1 = sp.Matrix([1, 2, 3])
    v2 = sp.Matrix([2, 4, 6])
    v3 = sp.Matrix([0, 1, 1])
    
    # Matrix with vectors as columns
    M = sp.Matrix.hstack(v1, v2, v3)
    print("Matrix M:\n", M)
    
    # Check linear dependence
    rank = M.rank()
    print("\nRank:", rank)
    print("Number of vectors:", M.cols)
    
    if rank < M.cols:
        print("=> Vectors are linearly dependent")
    else:
        print("=> Vectors are linearly independent")
    
    # Example linear combination c1*v1 + c2*v2 + c3*v3
    c1, c2, c3 = sp.symbols('c1 c2 c3')
    lin_comb = c1*v1 + c2*v2 + c3*v3
    print("\nLinear Combination: c1*v1 + c2*v2 + c3*v3 =")
    print(lin_comb)
    
    # Transition Matrix (Pick 3 independent vectors)
    # v2 is dependent, so use v1, v3 and one standard vector e1
    e1 = sp.Matrix([1,0,0])
    B = sp.Matrix.hstack(v1, v3, e1)   # A valid basis for R3
    
    print("\nBasis matrix B:\n", B)
    
    # Change of basis matrix and its inverse
    P = B
    P_inv = P.inv()
    
    print("\nChange-of-basis matrix P (B → Standard):\n", P)
    print("\nInverse P_inv (Standard → B):\n", P_inv)

    
    
# Q8. Find the orthonormal basis of a given vector space using the Gram-Schmidt orthogonalization process.
Input (code)

    import sympy as sp
    
    vectors = [sp.Matrix([1, 1, 0]), sp.Matrix([1, 0, 1]), sp.Matrix([0, 1, 1])]
    M = sp.Matrix.hstack(*vectors)
    print("Original vectors as columns:\n", M)
    
    Q, R = M.QRdecomposition()   # sympy QR gives orthogonal Q and upper-triangular R
    print("Orthogonal Q matrix:\n", Q)
    # Normalize columns of Q to unit vectors to ensure orthonormal (sympy Q is orthonormal if using proper QR)
    # Show orthonormal basis (columns of Q)
    orthonormal = [sp.Matrix(Q.col(i)) for i in range(Q.shape[1])]
    print("Orthonormal basis (columns of Q):")
    for vec in orthonormal:
        print(vec, "norm =", sp.sqrt(vec.dot(vec)))
# Q9. Check the diagonalizable property of matrices and find the corresponding eigenvalue and verify the Cayley-Hamilton theorem.
Input (code)

    import sympy as sp
    
    E = sp.Matrix([[5, 4, 2],
                   [0, 1, 0],
                   [0, 0, 3]])
    print("Matrix E:\n", E)
    
    eigs = E.eigenvects()
    print("Eigenvalues and eigenvectors:", eigs)
    
    # Check diagonalizable: number of independent eigenvectors == size
    is_diag = E.is_diagonalizable()
    print("Diagonalizable?", is_diag)
    
    # If diagonalizable, get P and D
    if is_diag:
        P, D = E.diagonalize()
        print("P (eigenvectors):\n", P)
        print("D (diagonal):\n", D)
    
    # Cayley-Hamilton: polynomial p(E) = 0 matrix for characteristic polynomial p
    x = sp.symbols('x')
    charpoly = E.charpoly(x).as_expr()
    print("Characteristic polynomial p(x):", charpoly)
    pE = sp.expand(charpoly.subs(x, E))
    print("p(E) is:\n", pE)   # should be zero matrix
# Q10. Application of Linear algebra: Coding and decoding of messages using nonsingular matrices. e.g., code “Linear Algebra is fun” and then decode it.
Input (code)

    import numpy as np
    # simple helper: convert text to numbers and back (A=0..Z=25)
    def text_to_nums(s):
        s = ''.join(ch for ch in s.upper() if ch.isalpha())
        return [ord(ch)-65 for ch in s]
    
    def nums_to_text(nums):
        return ''.join(chr((n % 26)+65) for n in nums)
    
    # message and 2x2 key (must be invertible mod26)
    msg = "Linear Algebra is fun"
    nums = text_to_nums(msg)
    # pad to even length for 2x2 block
    if len(nums) % 2:
        nums.append(23)   # pad with X
    
    K = np.array([[3, 3],
                  [2, 5]])   # determinant 3*5 - 3*2 = 9 => gcd(9,26)=1 invertible
    print("Key K:\n", K)
    
    # encode in blocks
    encoded = []
    for i in range(0, len(nums), 2):
        block = np.array(nums[i:i+2])
        cipher_block = (K.dot(block) % 26)
        encoded.extend(cipher_block.tolist())
    
    ciphertext = nums_to_text(encoded)
    print("Ciphertext:", ciphertext)
    
    # decode: need inverse of K mod 26
    det = int(round(np.linalg.det(K)))
    # modular inverse of determinant mod 26
    def egcd(a,b):
        if b==0: return (1,0,a)
        x,y,g = egcd(b, a%b)
        return (y, x - (a//b)*y, g)
    inv_det = None
    _, inv_det_candidate, g = egcd(det, 26)
    if g==1:
        inv_det = inv_det_candidate % 26
    # compute adjoint/inverse mod26
    adj = np.array([[K[1,1], -K[0,1]], [-K[1,0], K[0,0]]])
    K_inv_mod26 = (inv_det * adj) % 26
    # decode blocks
    decoded = []
    for i in range(0, len(encoded), 2):
        block = np.array(encoded[i:i+2])
        plain_block = (K_inv_mod26.dot(block) % 26)
        decoded.extend(plain_block.tolist())
    decoded_text = nums_to_text(decoded)
    print("Decoded text:", decoded_text)
# Q11. Compute Gradient of a scalar field.
Input (code)
    
    import sympy as sp
    
    x, y, z = sp.symbols('x y z', real=True)
    f = x**2 * y + sp.sin(y*z) + sp.exp(x*z)
    grad_f = [sp.diff(f, var) for var in (x, y, z)]
    print("Scalar field f(x,y,z):", f)
    print("Gradient ∇f =", sp.Matrix(grad_f))
# Q12. Compute Divergence of a vector field.
Input (code)

    import sympy as sp
    
    x, y, z = sp.symbols('x y z')
    P = x*y**2
    Q = sp.sin(x*z)
    R = sp.exp(y*z)
    divF = sp.diff(P, x) + sp.diff(Q, y) + sp.diff(R, z)
    print("Vector field F = (P,Q,R):", (P, Q, R))
    print("Divergence ∇·F =", sp.simplify(divF))
# Q13. Compute Curl of a vector field.
Input (code)

    import sympy as sp
    
    x, y, z = sp.symbols('x y z')
    P = y*z
    Q = x*z
    R = x*y
    
    # Compute curl = (R_y - Q_z, P_z - R_x, Q_x - P_y)
    curl = sp.Matrix([sp.diff(R, y) - sp.diff(Q, z),
                      sp.diff(P, z) - sp.diff(R, x),
                      sp.diff(Q, x) - sp.diff(P, y)])
    print("Vector field F:", (P, Q, R))
    print("Curl ∇×F =", curl)
















      




 
