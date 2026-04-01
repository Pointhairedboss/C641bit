# Neural SID - C64 Build
# Requires: ACME cross-assembler (https://sourceforge.net/projects/acme-crossass/)

ACME = acme
ACME_FLAGS = -f cbm -o
VICE = x64

PRG = neural-sid.prg
SRC = main.asm inference.asm sid_output.asm fastload.asm

.PHONY: all clean run

all: $(PRG)

$(PRG): $(SRC)
	$(ACME) $(ACME_FLAGS) $(PRG) main.asm

clean:
	rm -f $(PRG)

run: $(PRG)
	$(VICE) -autostart $(PRG)
