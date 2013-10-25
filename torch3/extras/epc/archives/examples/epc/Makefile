#
# Torch user Makefile example
#

# Torch location.
# Make sure to specify correct location...

TORCHDIR := $(shell cd ../..; pwd)

#
# All that follows you can probably keep as is...
#

OS := $(shell uname -s)
include $(TORCHDIR)/Makefile_options_$(OS)

%: %.cc
	cd $(TORCHDIR); ${MAKE}
	mkdir -p $(VERSION_KEY)
	$(CC) $(CFLAGS_$(MODE)) $(INCS) -o $(VERSION_KEY)/$@ $< $(LIBS)
