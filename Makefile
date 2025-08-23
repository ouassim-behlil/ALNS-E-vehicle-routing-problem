# Makefile to run EVRP solver and produce a plot
# Usage:
#   make all INSTANCE=data/E-n29-k4-s7.evrp ITER=200
#   make solve INSTANCE=... ITER=...  # only solve
#   make plot SOL=output/solution_...json  # only plot

INSTANCE ?= data/E-n29-k4-s7.evrp
ITER ?= 2000
PY ?= /home/ouassim/Desktop/ALNS-E-vehicle-routing-problem/evrp-venv/bin/python

# derive base name for outputs
INST_BASENAME := $(shell basename $(INSTANCE) .evrp)
SOL := output/solution_$(INST_BASENAME).json
PLOT := output/solution_$(INST_BASENAME).png

.PHONY: all solve plot clean
all: $(PLOT)

# run solver as a module so relative imports work
$(SOL): $(INSTANCE)
	@echo "Running solver on $(INSTANCE) (ITER=$(ITER)) -> $(SOL)"
	@mkdir -p output
	$(PY) -m src.file_solver $(INSTANCE) --iterations $(ITER) --output $(SOL)


# file rule to produce PNG plot from solution JSON
$(PLOT): $(SOL)
	@echo "Plotting solution $(SOL) -> $(PLOT)"
	$(PY) src/plot.py $(SOL) --out $(PLOT)

# phony alias
plot: $(PLOT)
	@echo "Plot created: $(PLOT)"

clean:
	rm -f output/solution_*.json output/solution_*.png
	@echo "cleaned output files"

solve: $(SOL)
	@echo "Solution written to $(SOL)"
