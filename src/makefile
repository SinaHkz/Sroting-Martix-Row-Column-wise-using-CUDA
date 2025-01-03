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
	ncu --metrics \
	l1tex__t_bytes.sum.per_second,\
	dram__bytes.sum.per_second,\
	gpu__time_duration.sum,\
	dram__sectors.sum,\
	sm__cycles_active.avg,\
	sm__throughput.avg.pct_of_peak_sustained_elapsed,\
	gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
	smsp__inst_executed.sum,\
	smsp__inst_executed.avg.per_cycle_active,\
	l1tex__t_requests.sum,\
	l1tex__t_sectors_hit.sum \
	$(EXEC) $(INPUT) $(OUTPUT)

profileImp:
	ncu --metrics \
	gpu__time_duration.sum,\
	dram__bytes.sum.per_second,\
	smsp__warps_active.avg.pct_of_peak_sustained_elapsed,\
	smsp__inst_executed.avg.per_cycle_active,\
	l1tex__t_sectors_hit.sum,\
	l1tex__t_requests.sum \
	../tests/main ../data/input.txt ../data/output.txt


profile2:
	nsys profile --trace=cuda --output=program_trace $(EXEC) $(INPUT) $(OUTPUT)

profileTime:
	nvprof $(EXEC) $(INPUT) $(OUTPUT)



clean:
	rm -f $(EXEC)

.PHONY: all run profile clean
