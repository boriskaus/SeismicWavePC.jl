#!/bin/bash
# Run a command and kill it when its memory exceeds a limit (protects the laptop from memory exhaustion: 2026-11-28 the
# machine panicked after four Julia jobs reached ~170 GB on 64 GB).
#   run_guarded.sh <limit_GB> <logfile> <command...>
# Every second: resident memory (ps, fast); every ~5 s also the footprint incl. compressed pages (top, slower; ps misses
# compressed pages). The larger value is compared with the limit and appended to <logfile>.mem.
limit_kb=$(( $1 * 1024 * 1024 )); log=$2; shift 2
"$@" > "$log" 2>&1 &
pid=$!
peak=0; foot=0; n=0
while kill -0 $pid 2>/dev/null; do
    procs="$pid $(pgrep -P $pid | tr '\n' ' ')"
    rss=$(ps -o rss= -p "$(echo $procs | tr ' ' ',')" 2>/dev/null | awk '{s+=$1} END {printf "%d", s}')
    if (( n % 5 == 0 )); then
        foot=$(top -l 1 -stats mem $(printf -- "-pid %s " $procs) | awk 'NR>1 && /^[0-9.]+[BKMG]/ {u=substr($1,length($1));
               v=substr($1,1,length($1)-1)+0; if (u=="G") v*=1048576; else if (u=="M") v*=1024; else if (u=="B") v/=1024; s+=v}
               END {printf "%d", s}')
    fi
    mem=$(( rss > foot ? rss : foot ))
    (( mem > peak )) && peak=$mem
    echo "$(date +%H:%M:%S) $(( mem / 1024 )) MB" >> "$log.mem"
    if (( mem >= limit_kb )); then
        echo "run_guarded: memory $(( mem / 1024 )) MB >= limit, killing $pid" | tee -a "$log"
        kill $procs 2>/dev/null; sleep 2; kill -9 $procs 2>/dev/null
        exit 99
    fi
    n=$(( n + 1 ))
    sleep 1
done
wait $pid; status=$?
echo "run_guarded: exit $status, peak memory $(( peak / 1024 )) MB" | tee -a "$log"
exit $status
