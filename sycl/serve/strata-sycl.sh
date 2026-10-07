#!/usr/bin/env bash
# strata-sycl.sh: the SYCL-built engine as a drop-in `exe` for serve/server.py (--engine strata). The port's binary
# needs the oneAPI runtime of the dev image, so this runs it there with stdin/stdout attached (the serve protocol is
# lines on those pipes) and stderr going to the server's log. Paths in the config's args are the container's:
# the data root is mounted at /work.
#   STRATA_SYCL_ROOT   host directory mounted at /work        (default: two levels above this script's repo)
#   STRATA_SYCL_IMAGE  the runtime image                      (default: strata-sycl-dev)
#   STRATA_SYCL_BIN    the engine binary, relative to the repo (default: build-sycl-aot/strata)
#   STRATA_SYCL_NAME   the container's name                   (default: strata-sycl-serve)
#   ONEAPI_DEVICE_SELECTOR  passed in when set; otherwise level_zero:gpu (every card) for a --layer-split, and the
#                      image's level_zero:0 for one card (#423)
#   STRATA_SYCL_RUNNER docker (the image, default); native - already inside a toolbox with oneAPI where the data root
#                      is /work (the server runs there too); distrobox:<name> - from the host, in that distrobox
# Passed into the container: every variable of this script's environment that starts with STRATA_ (the config's
# "env" block arrives that way: the server puts it in the engine's environment), ONEAPI_ (level_zero:* for a
# two-card split, #423), UR_, IGC_, SYCL_ or ZES_. The port's own defaults (below) are overridden the same way,
# e.g. "env": {"STRATA_VERIFY_NO_HOST": "0"} in the config.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
root=${STRATA_SYCL_ROOT:-$(dirname "$here")}
repo_in=/work/$(basename "$here")
name=${STRATA_SYCL_NAME:-strata-sycl-serve}
args=""
split=""
for a in "$@"; do
    args+=" $(printf '%q' "$a")"
    if [ "$a" = --layer-split ]; then split=1; fi
done
selector=${ONEAPI_DEVICE_SELECTOR:-${split:+level_zero:gpu}}
# the port's run-time switches: the device-built verify plan, and no host handshakes in it. NO_HOST is for a card on
# the xe driver (the host<->GPU flag stores are not visible there); an i915 card (Arc A-series) runs without it
# (docs/INTEL.md)
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_VERIFY_NO_HOST]=1 [STRATA_STAGER_THREADS]=12)
[ -n "$selector" ] && setting[ONEAPI_DEVICE_SELECTOR]=$selector
while IFS= read -r k; do
    case "$k" in
        STRATA_SYCL_*) ;;                                 # this script's own settings
        STRATA_*|ONEAPI_*|UR_*|IGC_*|SYCL_*|ZES_*|NEOReadDebugKeys|OverrideDefaultFP64Settings) setting[$k]=${!k} ;;
    esac
done < <(compgen -e)
envs=()
env=""
for k in "${!setting[@]}"; do
    # the engine tests the switches by presence: a value of 0 (or empty) means "not set", so it is not passed on
    case "${setting[$k]}" in 0|"") ;; *) envs+=(-e "$k=${setting[$k]}"); env+=" $k=$(printf '%q' "${setting[$k]}")" ;; esac
done
[ -n "${setting[ONEAPI_DEVICE_SELECTOR]:-}" ] || env+=" ONEAPI_DEVICE_SELECTOR=level_zero:0"
run="set +u; . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export$env; cd $repo_in && exec ${STRATA_SYCL_BIN:-build-sycl-aot/strata}$args"
case "${STRATA_SYCL_RUNNER:-docker}" in
native) exec bash -c "$run" ;;
distrobox:?*) exec distrobox enter "${STRATA_SYCL_RUNNER#distrobox:}" -- bash -c "$run" ;;
esac
docker rm -f "$name" >/dev/null 2>&1 || true              # a container left behind by a killed server
exec docker run --rm -i --name "$name" --device /dev/dri --oom-score-adj 1000 --stop-timeout 30 --no-healthcheck \
    -v "$root:/work" \
    "${envs[@]}" \
    "${STRATA_SYCL_IMAGE:-strata-sycl-dev}" \
    "cd $repo_in && exec ${STRATA_SYCL_BIN:-build-sycl-aot/strata}$args"
