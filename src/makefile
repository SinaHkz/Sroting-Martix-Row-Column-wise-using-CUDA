# Compiler
NVCC = nvcc

# Directories
SRC_DIR = .
OUT_DIR = ../tests
DATA_DIR = ../data

# Source files
SRC_FILES = main.cu utilities.cu kernels.cu
EXEC = $(OUT_DIR)/main
INPUT = $(DATA_DIR)/input.txt
OUTPUT = $(DATA_DIR)/output.txt

# Compilation flags
CFLAGS =

# Targets
all: $(EXEC)

$(EXEC): $(SRC_FILES)
	$(NVCC) $(SRC_FILES) -o $(EXEC)

run: $(EXEC)
	$(EXEC) $(INPUT) $(OUTPUT)

profile: $(EXEC)
	ncu --metrics l1tex__t_bytes.sum.per_second,dram__bytes.sum.per_second  $(EXEC) $(INPUT) $(OUTPUT)

clean:
	rm -f $(EXEC)

.PHONY: all run profile clean
