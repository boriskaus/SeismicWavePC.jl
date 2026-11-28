#!/bin/bash
# Run a command and kill it when its resident memory exceeds a limit (protects the laptop from memory exhaustion).
#   run_guarded.sh <limit_GB> <logfile> <command...>
# The RSS of the process tree is sampled every 5 s and appended to <logfile>.mem.
limit_kb=$(( $1 * 1024 * 1024 )); log=$2; shift 2
"$@" > "$log" 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
    # memory footprint incl. compressed pages (ps' RSS misses those: they filled up in the 2026-11-28 crash), in kB
    pids=$(printf -- "-pid %s " $pid $(pgrep -P $pid))
    rss=$(top -l 1 -stats mem $pids | awk 'NR>1 && /^[0-9.]+[BKMG]/ {u=substr($1,length($1)); v=substr($1,1,length($1)-1)+0;
          if (u=="G") v*=1048576; else if (u=="M") v*=1024; else if (u=="B") v/=1024; s+=v} END {printf "%d", s}')
    (( rss > peak )) && peak=$rss
    echo "$(date +%H:%M:%S) $(( rss / 1024 )) MB" >> "$log.mem"
    if (( rss > limit_kb )); then
        echo "run_guarded: RSS $(( rss / 1024 )) MB > limit, killing $pid" | tee -a "$log"
        pkill -P $pid; kill $pid; sleep 2; kill -9 $pid 2>/dev/null
        exit 99
    fi
    sleep 5
done
wait $pid; status=$?
echo "run_guarded: exit $status, peak RSS $(( peak / 1024 )) MB" | tee -a "$log"
exit $status
