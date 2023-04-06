# 1st abundans
abund=5
# dense grid (units of 10^10 cm^-3)
denses="5 10 15 25 40 65 105 170"
# Height grid (in Rjup)
hts="0.4 0.5 0.6 0.7 0.8"

for ht in $hts; do
  for den in $denses; do
    echo $ht $den $abund >> '../../exoXtransit/param.txt'
    sherpa -b top_level.py
  done
done

# above can be copy, pasted and tweaked as necessary
