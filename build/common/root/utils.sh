#!/bin/bash

log_info() {
    echo "[info] $1" | ts '%Y-%m-%d %H:%M:%.S'
}

log_warn() {
    echo "[warn] $1" | ts '%Y-%m-%d %H:%M:%.S'
}

log_error() {
    echo "[error] $1" | ts '%Y-%m-%d %H:%M:%.S'
}
