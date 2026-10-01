#!/bin/zsh
# Full Oregon State LoRa Diff_Days_Indoor_Setup: IQ_*.dat + IQ_*.sigmf-meta (no fft files).
# Uses the server's folder listing (devices missing on the server are skipped),
# resumes partial files, waits while the drive is missing, checks every size.
B="https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup"
D="/Volumes/TOSHIBA EXT/lora_oregon/Indoor"
for day in 1 2 3 4 5; do
  devs=$(curl -sk --max-time 60 "$B/Day$day/" | grep -oE 'href="Device[0-9]+/"' | grep -oE 'Device[0-9]+')
  for dev in ${=devs}; do
    files=$(curl -sk --max-time 60 "$B/Day$day/$dev/" | grep -oE 'href="IQ_[0-9]+\.(dat|sigmf-meta)"' | sed -E 's/href="([^"]+)"/\1/')
    for f in ${=files}; do
      while [ ! -d "/Volumes/TOSHIBA EXT/lora_oregon" ]; do echo "$(date) drive missing, waiting"; sleep 60; done
      mkdir -p "$D/Day$day/$dev"; o="$D/Day$day/$dev/$f"
      want=$(curl -skI --max-time 60 "$B/Day$day/$dev/$f" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
      have=$( [ -f "$o" ] && stat -f %z "$o" || echo 0 )
      if [ -n "$want" ] && [ "$have" = "$want" ]; then continue; fi
      for try in 1 2 3 4 5; do
        curl -sk -C - --retry 5 -o "$o" "$B/Day$day/$dev/$f"
        have=$(stat -f %z "$o" 2>/dev/null || echo 0)
        [ "$have" = "$want" ] && break
        while [ ! -d "/Volumes/TOSHIBA EXT/lora_oregon" ]; do sleep 60; done
      done
      if [ "$have" = "$want" ]; then echo "ok   Day$day/$dev/$f $have"; else echo "FAIL Day$day/$dev/$f have=$have want=$want"; fi
    done
  done
done
echo "=== INDOOR DOWNLOAD DONE $(date)"
