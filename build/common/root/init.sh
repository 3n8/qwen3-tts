#!/bin/bash

set -e

if [[ $(id -u) -eq 0 ]]; then
    echo "ERROR: Container must run as non-root user (current uid=$(id -u)), use user: directive in docker-compose" >&2
    exit 1
fi

exec 3>&1 4>&2 &> >(tee -a /config/supervisord.log)

source '/usr/local/bin/system/scripts/docker/utils.sh'

cat << "EOF"

    _ __      __             
   / |_/__   / /  ___   _____
  | |/|\ \ / /  / -_) / __/ |
  |___/ \_\_/_/  \__/ /_/   /_/
  https://github.com/3n8/qwen3-tts

EOF

source '/etc/image-build-info'

if [[ -z "${TZ}" ]]; then
    export TZ="UTC"
fi

if [[ -f "/usr/share/zoneinfo/${TZ}" ]]; then
    ln -sf "/usr/share/zoneinfo/${TZ}" /etc/localtime 2>/dev/null || true
fi

echo "[info] Timezone set to '${TZ}'" | ts '%Y-%m-%d %H:%M:%.S'
echo "[info] System information: $(uname -a)" | ts '%Y-%m-%d %H:%M:%.S'
echo "[info] Image architecture: '${TARGETARCH}'" | ts '%Y-%m-%d %H:%M:%.S'

export PUID=$(id -u)
export PGID=$(id -g)

echo "[info] Running as UID='${PUID}', GID='${PGID}'" | ts '%Y-%m-%d %H:%M:%.S'

echo "[info] Setting up directories..." | ts '%Y-%m-%d %H:%M:%.S'

for dir in /models /voices /out /root/.cache/huggingface; do
    if [[ -d "$dir" ]]; then
        if [[ -w "$dir" ]]; then
            echo "[info] Directory $dir is writable" | ts '%Y-%m-%d %H:%M:%.S'
        else
            echo "[warn] Directory $dir is not writable by current user" | ts '%Y-%m-%d %H:%M:%.S'
        fi
    else
        echo "[warn] Directory $dir does not exist, creating..." | ts '%Y-%m-%d %H:%M:%.S'
        mkdir -p "$dir" 2>/dev/null || echo "[warn] Could not create $dir" | ts '%Y-%m-%d %H:%M:%.S'
    fi
done

set +e
mkdir -p /config/run 2>/dev/null
chmod 775 /config/run 2>/dev/null
set -e

echo "[info] GPU configuration:" | ts '%Y-%m-%d %H:%M:%.S'
echo "[info]   HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" | ts '%Y-%m-%d %H:%M:%.S'
echo "[info]   HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-<not set>}" | ts '%Y-%m-%d %H:%M:%.S'

if command -v rocminfo &> /dev/null; then
    echo "[info] ROCm info:" | ts '%Y-%m-%d %H:%M:%.S'
    rocminfo 2>/dev/null | head -20 | while read line; do
        echo "    $line" | ts '%Y-%m-%d %H:%M:%.S'
    done
fi

echo "[info] Starting Supervisor..." | ts '%Y-%m-%d %H:%M:%.S'

exec 1>&3 2>&4

exec /usr/bin/supervisord -c /etc/supervisord.conf -n
