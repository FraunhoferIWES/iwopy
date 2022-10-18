# IWOPY Example: Electrostatics

This example finds a minimum of the electrostatic potential of `N` particles with unit charge, constraint to a radius of size `r`. 

## Solve with pygmo.IPOPT

For default radius and N = 10 particles, run
```
python run_pygmo_ipopt.py -n 10
```

Check the options by adding the `-h` flag.

## Solve with pygmo.NLOPT.CCSAQ

For default radius and N = 10 particles, run
```
python run_pygmo_nlopt.py -n 10 -a ccsaq
```
Other algorithms from `NLOPT` are available using different choices for `-a` (or `--opt_algo`).

Check the options by adding the `-h` flag.

## Solve with pymoo.GA

For default radius and N = 20 particles, run
```
python run_pymoo.py -n 20 -a ga
```

Check the options by adding the `-h` flag.

## Solve with pymoo.PSO

For default radius and N = 20 particles, run
```
python run_pymoo.py -n 20 -a pso 
```

Check the options by adding the `-h` flag.
