To run a single iteration use a ciao terminal active in the code/ directory with the command:
    sherpa -b top_level.py
Or, in the case of running as part of a loop to explore parameter space, run this 
with ciao & bash activated:
    for sclht in 0 1 2 3 4; do for dense in 0 1 2 3 4 5 6; do for abund in 0 1 2 3 4 5 6; 
    do echo $sclht $dense $abund >> param.txt; sherpa -b top_level.py;  done; done; done
The sets of numbers are the indicies for the arrays at the start of top level.py and 
can be adjusted to match those arrays as needed.

Note: in current version a single gridpoint (sclht, dense, abund) takes ~2.5 hrs to run. I'll 
be working to improve both this and the documentation with changes made available periodically.

Dependencies:
    Most likely included with ciao & sherpa:
        numpy
        scipy
        matplotlib 
        pycrates
        sherpa
        pandas
        astropy
        pickle
    Will likely need to be installed with pip3 install:
        tqdm (to see progress bar)
        mendeleev (for element masses)
        glymur (read .jp2 (JPEG 2000) files for AIA images -- need openJPEG > 2.3.0)
        glob (to locate lists of files with matching formats using the wildcard (*))
        celluloid (to create animations of the plots as desired)

Any questions please email me to let me know.
