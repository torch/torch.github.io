---
id: developer-docs
title: Developer Documentation
layout: docs
permalink: /docs/developer-docs.html
previous: package-docs.html
next: tutorials.html
---

Writing your own nn modules
===========================

We will see how to create your own new modules, and testing them.
You should be able to plug them into existing neural networks seamlessly.

If the modules are simple and not performance-critical,
then you can simply write them in a few lines of Lua (Section 1).

If the module is heavier in computation, or you want to create
specialized and optimized code for CPU or GPU, you might want to
create the modules at the C / CUDA level (Section 2).

> Modules are bricks to build neural networks. A Module is a neural network by itself, but it can be combined with other networks using container classes to create complex neural networks. Module is an abstract class which defines fundamental methods necessary for a training a neural network. All modules are serializable.

Modules contain two states variables: `output` and `gradInput`.
Here we review the set of basic functions that a `Module` has to implement:

```lua
[output] forward(input)
```

Takes an input object, and computes the corresponding output of the module.
In general input and output are Tensors. However, some special
sub-classes like table layers might expect something else.
Please, refer to each module specification for further information.

After a `forward()`, the `output` state variable should have been updated to the new value.

It is not advised to override this function. Instead, one should
implement `updateOutput(input)` function.
The `forward(input)` function in the abstract parent class `Module` will call `updateOutput(input)`.

```lua
[gradInput] backward(input, gradOutput)
```

Performs a back-propagation step through the module, with respect to the given input.
In general this method makes the assumption `forward(input)` has been called before, with the same input.
This is necessary for optimization reasons.

> If you do not respect this rule, `backward()` will compute incorrect gradients.


In general `input`, `gradOutput` and `gradInput` are `Tensors`.
However, some special sub-classes like `table` layers might expect something else.
Please, refer to each module specification for further information.

A backpropagation step consist in computing two kind of gradients at input
given `gradOutput` (gradients with respect to the output of the module).

This function simply performs this task using two function calls:

- A function call to `updateGradInput(input, gradOutput)`.
- A function call to `accGradParameters(input,gradOutput)`.

It is not advised to override this function call in custom classes.
It is better to override `updateGradInput(input, gradOutput)` and `accGradParameters(input, gradOutput)` functions.

```lua
[output] updateOutput(input)
```

When defining a new module, this method should be overloaded.

Computes the output using the current parameter set of the class and input.
This function returns the result which is stored in the `output` field.

```lua
[gradInput] updateGradInput(input, gradOutput)
```

When defining a new module, this method should be overloaded.

Computing the gradient of the module with respect to its own `input`.
This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

```lua
accGradParameters(input, gradOutput)
```

When defining a new module, this method may need to be overloaded, if the module has trainable parameters.

Computing the gradient of the module with respect to its own parameters.
Many modules do not perform this step as they do not have any parameters.
The state variable name for the parameters is module dependent.
The module is expected to accumulate the gradients with respect to the parameters in some variable.

Zeroing this accumulation is achieved with `zeroGradParameters()` and
updating the parameters according to this accumulation is done with `updateParameters()`.

```lua
reset()
```

This method defines how the trainable parameters are reset, i.e. initialized before training.

Modules provide a few other methods that you might want to define,
if you are not planning to use the `optim` package.
These methods help `zero()` the parameters, and update them using very basic techniques.

In terms of code structure, Torch provides a class model, which we use for inheritance,
and in general for the definition of all the modules in nn.

Here is an empty holder for a typical new class:

```lua
local NewClass, Parent = torch.class('nn.NewClass', 'nn.Module')

function NewClass:__init()
   Parent.__init(self)
   end

function NewClass:updateOutput(input)
end

function NewClass:updateGradInput(input, gradOutput)
end

function NewClass:accGradParameters(input, gradOutput)
end

function NewClass:reset()
end
```

When defining a new class, all we need to do is fill in these empty functions.
Note that when defining the constructor `__init()`, we always call the parent's constructor first.

Let's see some practical examples now.


1. Writing modules at the Lua level: Implementing Dropout Activation Units
=======================================================================

Dropout units have a central idea there is to perturbate the activations of hidden units, by randomly zeroing some of these units.

Such a class could be defined like this:

```lua
local Dropout, Parent = torch.class('nn.Dropout', 'nn.Module')

function Dropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

function Dropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.noise:resizeAs(input)
   self.noise:bernoulli(1-self.p)
   self.output:cmul(self.noise)
   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   return self.gradInput
end
```

When writing modules with gradient estimation, it's always very important to test your implementation. This can be easily done using the `Jacobian` class provided in `nn`, which compares the implementation of the gradient methods (`updateGradInput()` and `accGradParameters()`) with the `Jacobian` matrix obtained by finite differences (perturbating the input of the module, and estimating the deltas on the output). This can be done like this:

```lua
-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 0.5
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.Dropout(percentage)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end
```

One slight issue with the `Jacobian` class is the fact that it assumes that the outputs of a module are deterministic wrt to the inputs. This is not the case for that particular module, so for the purpose of these tests we need to freeze the noise generation, i.e. do it only once:

-- we overload the updateOutput() function to generate noise only
-- once for the whole test.

```lua
function Dropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.noise = self.noise or input.new():resizeAs(input):bernoulli(1-self.p)
   self.output:cmul(self.noise)
   return self.output
end
```

2. Writing modules at the C or CUDA level
=========================================

## C macro based templates

Before writing Torch C code, one has to first get familiar with the C macro syntax that is sprinkled all over Torch and nn.

For example, look at this code that appears in [THTensorMath.c](https://github.com/torch/torch7/blob/c55a0621ae5f306fcd4edf03bd382dd3729972d9/lib/TH/generic/THTensorMath.c#L374-L388)

```C
void THTensor_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] + value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}
```

The strange `_(add)(THTensor *r_ ....)` syntax that you see is a preprocessor macro.

```C
lib/TH/THTensor.h:
#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
```
which leads to...

```C
lib/TH/THGeneral.h.in:
#define TH_CONCAT_4(x,y,z,w) TH_CONCAT_4_EXPAND(x,y,z,w)
```
and finally...

```C
lib/TH/THGeneral.h.in:
#define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
```

Therefore, (and after preprocessing with a few more macros),

```C
void THTensor_(add)(THTensor *r_, THTensor *t, real value)
```

ultimately becomes this:

```C
long THRealTensor_add(const THRealTensor *r_, THRealTensor *t, real value)
```

Real and real are defined to be of a specific type, for example, for float precision:

```C
#define Real Float
#define real float
```

finally makes that function prototype:

```C
long THFloatTensor_add(const THFloatTensor *r_, THFloatTensor *t, float value)
```

[Aren't preprocessors just grand ?](http://stackoverflow.com/questions/30420807/strange-c-syntax-in-lua-library)

You will see similar syntax in the nn library as well, so brace yourself well for this syntax.


## C nn module

The best way to understand how to write a new nn module is by looking at an existing one.

Here is a walkthrough of writing nn.Threshold:

### Step 1: write the Lua part

https://github.com/torch/nn/blob/master/Threshold.lua

- write the constructor
- write updateOutput / updateGradInput that simply call into C

Calling into C has an efficient but slightly weird syntax:

https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/Threshold.lua#L20

```lua
input.nn.Threshold_updateOutput(self, input)
```

This line is simply calling the function defined here:
https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/generic/Threshold.c#L5

And the reason it calls that is because you register the C function under the input.nn. table here: https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/generic/Threshold.c#L61-L63

This helps us write generic code that works for any defined tensor type, while not doing any complicated dynamic function dispatch logic.

The complete nn.Threshold module is written as two files:
- Lua part: https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/Threshold.lua
- C part: https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/generic/Threshold.c

The files are included into the package at these lines:
- init.lua : https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/init.lua#L68
- init.c : https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/init.c#L41-L42
- init.c : https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/init.c#L153
- init.c : https://github.com/torch/nn/blob/b80e26e8b849a69b8121acf62f3487095c2f10e8/init.c#L194

## CUDA nn module

To add CUDA support to the nn.Threshold module, similar to writing a Threshold.c, we will write a Threshold.cu

- https://github.com/torch/cunn/blob/master/lib/THCUNN/Threshold.cu

and include it here:

- https://github.com/torch/cunn/blob/master/init.cu#L36
- https://github.com/torch/cunn/blob/master/utils.h#L30

