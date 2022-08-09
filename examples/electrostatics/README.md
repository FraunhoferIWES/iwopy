# IWOPY Example: Electrostatics

This example finds a minimum of the electrostatic potential of `N` particles with unit charge, constraint to a radius of size `r`. 

## Solve with pygmo.IPOPT

For default radius and N = 4 particles, run
```
python run_pygmo_ipopt.py -n 4 --pop
```
The `--pop` is optional and might speed up the gradient calculation.

Check the options by adding the `-h` flag.

## Solve with pymoo.GA

For default radius and N = 20 particles, run
```
python run_pymoo.py -n 20 -a ga --pop
```
The `--pop` is optional and vecctorizes the population calculation.

Check the options by adding the `-h` flag.

## Solve with pymoo.PSO

For default radius and N = 20 particles, run
```
python run_pymoo.py -n 20 -a pso --pop
```
The `--pop` is optional and vecctorizes the population calculation.

Check the options by adding the `-h` flag.
