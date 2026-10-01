#!/bin/zsh
# first 2 s (16 MB) of IQ_1 and IQ_2, devices 1-10, days 1-5 (HTTP range requests; nothing else)
B="https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup"
for day in 1 2 3 4 5; do for dev in $(seq 1 10); do for f in 1 2; do
  o=head2s/d${day}_dev${dev}_IQ_${f}.dat
  [ -s $o ] && [ $(stat -f %z $o) -ge 16000000 ] && continue
  curl -sk --retry 5 -r 0-15999999 -o $o "$B/Day$day/Device$dev/IQ_$f.dat"; echo "$o $(stat -f %z $o)"
done; done; done
echo FETCH DONE
