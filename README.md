# wfcld
Wavefunction collapsing, yay!

`main2.py` is the one you should use. `main.py` was using a one-tiled model, and it didn't even work. I re-did the whole thing in `main2.py`, and it shall replace main.py in the near future.

Todo:
* Test if it's really working (update: maybe it actually works?)
* Data visualisation in a GUI : every blocks and every possible neighbors + probabilities (simplify debugging a lot)

Done:
* Right entropy computation -> What is right, what is wrong, who am I?
* Adds probas
* 4-connected instead of 8-connected (branch) -> I didn't branch. Woops.
* Takes N\*N input
