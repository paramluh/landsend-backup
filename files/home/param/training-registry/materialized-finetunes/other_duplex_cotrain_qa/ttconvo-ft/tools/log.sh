# source me: log "msg" appends a PT-stamped line to the staging log
log(){ echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z') [$(hostname)] $*" | tee -a /data/home/amartyavarma/ppm/ttconvo-ft/logs/staging.log; }
