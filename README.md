# Mathematics for Computing – Practical Programs  

---

## 🔢 **Q1. Create and transform vectors and matrices (Transpose, Conjugate & Hermitian)**

### **Input (Code)**

```python
import numpy as np
# horizontal vector from list
h = np.array([1+2j, 2-1j, 3.0])   
print("Horizontal vector h:", h)

# vertical vector (column)
v = h.reshape(-1, 1)
print("Vertical vector v (column):\n", v)

# matrix creation
A = np.array([
    [1, 2+1j, 3],
    [4, 5, 6-2j],
    [7, 8, 9]
])
print("Matrix A:\n", A)

# transpose
At = A.T
print("Transpose A^T:\n", At)

# conjugate transpose (Hermitian)
A_h = np.conjugate(A.T)
print("Conjugate transpose (A^H):\n", A_h)

