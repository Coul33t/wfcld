# wfcld
Wavefunction collapsing, yay!

`main2.py` is the one you should use. `main.py` was using a one-tiled model, and it didn't even work. I re-did the whole thing in `main2.py`, and it shall replace main.py in the near future.

Todo:
* Test if it's really working (update: maybe it actually works?)
* Check why there are a lot of contradictions (maybe related to the next point)
* Check why the array seems to be cyclic (look into entropy computation)
* Data visualisation in a GUI : every blocks and every possible neighbors + probabilities (simplify debugging a lot)

Done:
* Right entropy computation -> What is right, what is wrong, who am I?
* Add probas
* 4-connected instead of 8-connected (branch) -> I didn't branch. Woops.
* Take N\*N input -> Now taking NxM input
* Extensive testing of the submatrix matching algorithm (there's probably a problem coming from here)
