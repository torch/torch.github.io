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
- `eye()` which returns a identity matrix
- `*` operator over matrices (which performs a matrix-vector or matrix-matrix multiplication)

We first make sure the random seed is the same for everyone

```lua
torch.manualSeed(1234)
```

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
```

Printing the function value (here on a random point) can be easily done with:

```lua
print(J(torch.rand(N))
```

## 2. Find the exact mimimum


We can inverse the matrix (which might not be numerically optimal)

```lua
xs = torch.inverse(x)*b
print(string.format('J(x^*) = %g', J(xs)))
```

## 3. Search the minimum by gradient descent

We first define the gradient w.r.t. `x` of `J(x)`:

```lua
function dJ(x)
  return A*x-b
end
```

We then define some current solution:

```lua
x = torch.rand(N)
```

And then apply gradient descent (with a given learning rate `lr`) for a while:

```
lr = 0.01
for i=1,20000 do
  x = x - dJ(x)*lr
  -- we print the value of the objective function at each step
  print(string.format('at iter %d J(x) = %f', i, J(x)))
end
```

You should see

```
...
at iter 19995 J(x) = -3.135664
at iter 19996 J(x) = -3.135664
at iter 19997 J(x) = -3.135665
at iter 19998 J(x) = -3.135665
at iter 19999 J(x) = -3.135665
at iter 20000 J(x) = -3.135666
```

## 4. Using the optim package

First, you need to install the `optim` package:

```
luarocks install optim
```

#### A word on local variables

In practice, it is *never* a good idea to use global variables. Use `local` at
everywhere. In our examples, we have defined everything in global, such that
they can be cut-and-pasted in the interpreter command line.
Indeed, defining a local like:

```lua
local A = torch.rand(5, 5)
```

will be only available to the current scope, which, when running the interpreter, is limited
to the current input line. Subsequent lines would not have access to this local.

In lua one can define a scope with the `do...end` directives:

```lua
do
   local A = torch.rand(5, 5)
   print(A)
end
print(A)
```

If you cut-and-paste this in the command line, the first print will be a
5x5 matrix (because the local `A` is defined for the duration of the scope
`do...end`), but will be `nil` afterwards.

#### Defining a closure with an upvalue

We need to define a closure which returns both `J(x)` and `dJ(x)`.  Here we
define a scope with `do...end`, such that the local variable `iter` is an
upvalue to `JdJ(x)`: only `JdJ(x)` will be aware of it.  Note that in a
script, one would not need to have the `do...end` scope, as the scope of
`iter` would be until the end of the script file (and not the end of the
line like the command line).

```lua
do
   local iter = 0
   function JdJ(x)
      iter = iter + 1
      print(string.format('at iter %d J(x) = %f', iter, J(x)))
      return J(x), dJ(x)
   end
end
```

#### Training with optim

We first define a state for conjugate gradient:

```lua
state = {
   verbose = true,
   maxIter = 100
}
```

and now we train:

```lua
x = torch.rand(N)
optim.cg(JdJ, x, state)
```

You should see something like:

```
at iter 120 J(x) = -3.136835
at iter 121 J(x) = -3.136836
at iter 122 J(x) = -3.136837
at iter 123 J(x) = -3.136838
at iter 124 J(x) = -3.136840
at iter 125 J(x) = -3.136838
```
