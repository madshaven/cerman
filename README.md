

# Cerman
<!-- **Simulating streamer propagation in dielectric liquids** -->
> streamer breakdown, dielectric liquid, simulation model, python, computational physics
<!-- 
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
 -->

[Electrical breakdown](https://en.wikipedia.org/wiki/Electrical_breakdown)
in liquids is preceded by a breakdown channel called a _streamer_.
This software is developed to simulate 
[a model](https://dx.doi.org/10/cxjf) 
for the propagation of positive streamers in dielectric liquids. 
The name _Cerman_ is an abbreviation of 
[ceraunomacy](https://en.wiktionary.org/wiki/ceraunomancy).

References [[1-3](#references)] provide information on streamers and the model.
Reference [[4](#references)] provides information on the implementation of the model and its usage.


## Getting started

### Installation
- Note: The software is developed for OSX and Linux, and has not been tested on Windows.
- Install Python 3.6 or above - for instance from
[Python.org](https://www.python.org/downloads/)
or
[Anaconda](https://www.anaconda.com/products/individual#Downloads).
- If desired, create a virtual environment, see for example [`conda create`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).
- Clone (or download) the current [repo](https://github.com/madshaven/cerman).
- Install the package using `pip` from the appropriate folder. `pip` also installs the dependencies 
`numpy`,
`simplejson`,
`matplotlib`,
`scipy`,
and
`statsmodels` if needed.
```shell
$ git clone https://github.com/madshaven/cerman.git
$ cd cerman
$ pip install .  # installs cerman and dependencies
```
- The software provides the command `cerman`.
Verify the installation by running the command to display usage and actions.
```shell
$ cerman        # display usage
$ cerman help   # display actions
```


### Example files
- Each example input file, `example_name.json`, specifies a simulation series.
- Example files are located in `cerman/examples/example_name`.
- The `Makefile`, in the same folder, specifies how the example can be simulated and how some of the results can be plotted.
- The example files can be used to replicate results from 
the references [[1-4](#references)].


### Example of usage
Copy one of the example input files,
for instance `small_gap.json`
 to an appropriate project folder.
```shell
$ mkdir small_gap_test
$ cd small_gap_test
$ cp path_to_cerman/examples/small_gap/small_gap.json .
```
Create simulation input parameter files, 
in this case, 100 files named `small_gap_###.json`.
Plot the parameters, for instance for the voltage, as below.
```shell
$ # create input files
$ cerman ci -f small_gap.json
$ # plot input files
$ cerman pp -g "_0?1.json"
```
Start a series of simulations in loop, globbing for files matching a pattern.
```shell
$ # simulate in sequential loop
$ cerman sims -g "_0?0.json"
$ # simulate in parallel loop using 4 threads
$ cerman sims -g "_0?0.json" -m 4
```
Each simulation `small_gap_###.pkl` 
creates a corresponding save file `small_gap_###_stat.pkl`.
Plot simulation results saved in these files.
```shell
$ # plot streamer position vs simulated time
$ cerman ps streak -g _stat -o "legend=v"  # voltage as legend label
$ # plot streamer, xz and yz, 0.3 hspace between each series
$ cerman ps shadow -g _stat -o "legend=v diffx=.3"
$ # plot streamer position vs overall average speed
$ cerman ps speed -g _stat -o "legend=v xmax=2 method=average_v"
$ # plot streamer position vs windowed average speed
$ cerman ps speed -g _stat -o "legend=v xmax=2 method=window_v"
```
Parse simulation results in `small_gap_???_stat.pkl` to create `small_gap_stat.stat` 
and
plot the average propagation speed and final propagation length against voltage.
```shell
$ # parse results and compile archive
$ cerman ca -g _stat
$ # plot results from archive
$ cerman pr v -g _stat -o "ykeys=psa_ls"
```

  
## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 © Inge Madshaven.


## References

1. **Inge Madshaven, Per-Olof Åstrand, Øystein Leif Hestad, Stian Ingebrigtsen, Mikael Unge, Olof Hjortstam**\
_Simulation model for the propagation of second mode streamers
in dielectric liquids using the Townsend–Meek criterion_\
Journal of Physics Communications 2:105007 (2018)\
doi: [10/cxjf](https://dx.doi.org/10/cxjf) | arXiv: [1804.10473](https://arxiv.org/abs/1804.10473)
1. **Inge Madshaven, Øystein Leif Hestad, Mikael Unge, Olof Hjortstam, Per-Olof Åstrand**\
_Conductivity and capacitance of streamers in avalanche model
for streamer propagation in dielectric liquids_\
Plasma Research Express 1:035014 (2019)\
doi: [10/c933](https://dx.doi.org/10/c933) | arXiv: [1902.03945](https://arxiv.org/abs/1902.03945)
1. **Inge Madshaven, Øystein Leif Hestad, Mikael Unge, Olof Hjortstam, Per-Olof Åstrand**\
_Photoionization model for streamer propagation mode change
in simulation model for streamers in dielectric liquids_\
Plasma Research Express 2:015002 (2020)\
doi: [10/dg8m](https://dx.doi.org/10/dg8m) | arXiv: [1909.12694](https://arxiv.org/abs/1909.12694)
1. **Inge Madshaven, Øystein Leif Hestad, Per-Olof Åstrand**\
_Cerman: Software for simulating streamer propagation
in dielectric liquids based on the Townsend–Meek criterion_\
doi: [10/f9bw](https://dx.doi.org/10/f9bw) | arXiv: [2007.02999](https://arxiv.org/abs/2007.02999)
