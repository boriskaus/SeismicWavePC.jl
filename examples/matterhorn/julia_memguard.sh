#!/bin/bash
# Machine-wide guard against memory exhaustion by Julia jobs (2026-11-28: a kernel watchdog panic after four Julia
# processes reached ~170 GB footprint on a 64 GB laptop). Every 5 s it reads the memory footprint (incl. compressed
# pages) of all julia processes and the system's free-memory level. If the julia total exceeds MAX_GB, or the free level
# drops below MIN_FREE %, it terminates the largest julia process (SIGTERM, SIGKILL after 10 s), logs it to
# ~/Library/Logs/julia_memguard.log and shows a notification.
#
# Try it in a terminal:   bash julia_memguard.sh
# Install (runs at login):
#   cp julia_memguard.sh ~/bin/ && cp com.user.julia-memguard.plist ~/Library/LaunchAgents/ &&
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.julia-memguard.plist
# Remove:
#   launchctl bootout gui/$(id -u)/com.user.julia-memguard && rm ~/Library/LaunchAgents/com.user.julia-memguard.plist
MAX_GB=${JULIA_MEMGUARD_MAX_GB:-44}     # total footprint of all julia processes [GB]
MIN_FREE=${JULIA_MEMGUARD_MIN_FREE:-12} # kern.memorystatus_level (system-wide free memory, %)
LOG=~/Library/Logs/julia_memguard.log
while true; do
    lines=$(top -l 1 -o mem -stats pid,command,mem -n 40 | awk '$2=="julia" {m=$3; u=substr(m,length(m)); v=substr(m,1,length(m)-1)+0;
        if (u=="G") v*=1024; else if (u=="K") v/=1024; else if (u=="B") v/=1048576; printf "%d %d\n", $1, v}')
    total=$(echo "$lines" | awk '{s+=$2} END {print s+0}')
    free=$(sysctl -n kern.memorystatus_level)
    if [ -n "$lines" ] && { [ "$total" -gt $((MAX_GB * 1024)) ] || [ "$free" -lt "$MIN_FREE" ]; }; then
        read -r pid mb <<< "$(echo "$lines" | sort -k2 -n -r | head -1)"
        cmd=$(ps -o command= -p "$pid" | cut -c1-200)
        echo "$(date '+%F %T') julia total ${total} MB, free ${free}%: killing pid $pid (${mb} MB): $cmd" >> "$LOG"
        kill -TERM "$pid" 2>/dev/null
        osascript -e "display notification \"Killed julia pid $pid (${mb} MB)\" with title \"julia_memguard\"" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -KILL "$pid" 2>/dev/null
        sleep 5
    fi
    sleep 5
done
