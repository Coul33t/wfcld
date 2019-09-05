# wfcld
Wavefunction collapsing, yay!

`main2.py` is the one you should use. `main.py` was using a one-tiled model, and it didn't even work. I re-did the whole thing in `main2.py`, and it shall replace main.py in the near future.

### Required libraries:
* Numpy (computation, image related stuff)
* PIL (every image-related stuff)
* OpenCV (video export)

### Usage:
`py main2.py`

### Arguments:
* (optional) `-g` \ `--gif`: export the process to GIF. If there is no final image (it reached contradictions everytime), it displays the best image obtained (the ones with the less empty blocks)
* (optional) `-v` \ `--video`: same for video
* (optional) `-d` \ `--debug`: prints a lot of boring stuff in the console

### Output:
* The final image and possibly a gif and a video of the process. If there is no final image (it reached contradictions everytime), it displays the best image obtained (the ones with the less empty blocks)

Be aware that the quality of the GIF and the video isn't great at all for small images.
_________________________________________________________________________________
### Todo:
* Test if it's really working (update: maybe it actually works?)
* Check why there are a lot of contradictions (maybe related to the next point)
* Check why the array seems to be cyclic (look into entropy computation)
* Data visualisation in a GUI : every blocks and every possible neighbors + probabilities (simplify debugging a lot)

### Done:
* Right entropy computation -> What is right, what is wrong, who am I?
* Add probas
* 4-connected instead of 8-connected (branch) -> I didn't branch. Woops.
* Take N\*N input -> Now taking NxM input
* Extensive testing of the submatrix matching algorithm (there's probably a problem coming from here)
