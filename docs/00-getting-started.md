---
id: getting-started
title: Getting started with Torch
layout: docs
permalink: /docs/getting-started.html
next: five-simple-examples.html
---

## Installing Torch

We provide a simple installation process for Torch on Mac OS X and Ubuntu 12+:

Torch can be installed by running these two commands:

```bash
$ curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-luajit+torch | bash
```
The [first script](https://raw.githubusercontent.com/torch/ezinstall/master/install-deps) 
installs the basic package dependencies that LuaJIT and Torch require. 
The [second script](https://raw.githubusercontent.com/torch/ezinstall/master/install-luajit+torch) 
installs [LuaJIT](http://luajit.org/luajit.html), [LuaRocks](http://luarocks.org/), 
and then uses LuaRocks (the lua package manager) to install core packages like
[torch](https://github.com/torch/torch7/blob/master/README.md), 
[nn](https://github.com/torch/nn/blob/master/README.md) and 
[paths](https://github.com/torch/paths/blob/master/README.md), as well as a few other packages. 

New packages can be installed using Luarocks from the command-line:

```bash
$ luarocks install image
$ luarocks list
```

Once installed you can run torch with the command "th" from you prompt!

The easiest way to learn and experiment with Torch is by starting an
interactive session (also known as the torch read-eval-print loop or [TREPL](https://github.com/torch/trepl/blob/master/README.md)):

```bash
$ th
 
  ______             __   |  Torch7                                   
 /_  __/__  ________/ /   |  Scientific computing for Lua.         
  / / / _ \/ __/ __/ _ \  |                                           
 /_/  \___/_/  \__/_//_/  |  https://github.com/torch   
                          |  http://torch.ch            
			  
th> torch.Tensor{1,2,3}
 1
 2
 3
[torch.DoubleTensor of dimension 3]

th>
```

To exit the interactive session, type `^c` twice — the control key
together with the `c` key, twice, or type `os.exit()`.
Once the user has entered a complete expression, such as ``1 + 2``, and
hits enter, the interactive session evaluates the expression and shows
its value. 

To evaluate expressions written in a source file `file.lua`, write
`th> dofile "file.lua"`.

To run code in a file non-interactively, you can give it as the first
argument to the `th` command::

```bash
$ th file.lua
```

There are various ways to run Lua code and provide options, similar to
those available for the ``perl`` and ``ruby`` programs:

```bash
 $ th -h
Usage: th [options] [script.lua [arguments]]

Options:
  -l name            load library name
  -e statement       execute statement
  -h,--help          print this help
  -a,--async         preload async (libuv) and start async repl (BETA)
  -g,--globals       monitor global variables (print a warning on creation/access)
  -gg,--gglobals     monitor global variables (throw an error on creation/access)
  -x,--gfx           start gfx server and load gfx env
  -i,--interactive   enter the REPL after executing a script
```

TREPL is full of convenient features likes:

* Tab-completion on nested namespaces
* Tab-completion on disk files (when opening a string)
* History (preserved between sessions)
* Pretty print (table introspection and coloring)
* Auto-print after eval (can be stopped with ;)
* Each command is profiled, timing is reported
* No need for '=' to print
* Easy help with: `? funcname`
* Self help: `?`
* Shell commands with: $ cmd (example: `$ ls`)

### Next steps

In addition to this manual, there are various other resources that may
help new users get started with torch, all summarized in this [Cheatsheet](https://github.com/torch/torch7/wiki/Cheatsheet)
  
The cheatsheet provides links to tutorials, demos, package summaries and a lot of useful information.

If you have a question, please come join the [Torch users mailing list](https://groups.google.com/forum/embed/?place=forum/torch7#!forum/torch7)