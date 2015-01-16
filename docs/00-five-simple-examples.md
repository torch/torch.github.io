---
id: five-simple-examples
title: Five simple examples
layout: docs
permalink: /docs/five-simple-examples.html
prev: getting-started.html
---
## 1. Define a positive definite quadratic form

We rely on few torch functions here:
  - `rand()` which creates tensor drawn from uniform distribution
  - `t()` which transposes a tensor (note it returns a new view)
  - `dot()` which performs a dot product between two tensors
  - `*` operator over matrices (which performs a matrix-vector or matrix-matrix multiplication)

```lua
-- choose a dimension
N = 5

-- create a random NxN matrix
A = torch.rand(N, N)

-- make it symmetric positive
A = A*A:t()

-- make it definite
A:add(0.001, torch.eye(5))

-- add a linear term
b = torch.rand(5)

-- create the quadratic form
function J(x)
   return 0.5*x:dot(A*x)-b:dot(x)
end

-- evaluate and print a random point
print(J(torch.rand(N))
```

## 2. Search the minimum by gradient descent

