---
id: five-simple-examples
title: Five simple examples
layout: docs
permalink: /docs/five-simple-examples.html
prev: getting-started.html
---
## 1. Define a positive definite quadratic form

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

