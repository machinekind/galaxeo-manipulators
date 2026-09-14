# Sourced by the payloads after their declarations. Makes sure the LeRobot
# dataset at DATASET_ROOT is complete on this machine, and never lets the
# loader wander off to the Hub when it is not.
#
# The checkout is synced with exclude patterns that belong to whoever runs the
# sync, and a floor pattern such as *.mp4 silently strips a dataset's videos
# on the way. So the dataset also travels as one archive, DATASET_ARCHIVE,
# holding the root's last directory component; when the root is missing or
# incomplete and the archive is here, it is unpacked next to it.

dataset_complete() {
    [ -f "$1/meta/info.json" ] || return 1
    [ -n "$(find "$1/data" -name '*.parquet' -print -quit 2>/dev/null)" ] || return 1
    [ -n "$(find "$1/videos" -name '*.mp4' -print -quit 2>/dev/null)" ] || return 1
}

if ! dataset_complete "$DATASET_ROOT"; then
    if [ -n "${DATASET_ARCHIVE:-}" ] && [ -f "$DATASET_ARCHIVE" ]; then
        echo "dataset at $DATASET_ROOT is missing or incomplete; unpacking $DATASET_ARCHIVE"
        rm -rf "$DATASET_ROOT"
        mkdir -p "$(dirname "$DATASET_ROOT")"
        tar -C "$(dirname "$DATASET_ROOT")" -xf "$DATASET_ARCHIVE"
    fi
fi
if ! dataset_complete "$DATASET_ROOT"; then
    echo "ERROR: no complete dataset at $DATASET_ROOT (meta, parquet and mp4 all needed)" >&2
    exit 1
fi
echo "dataset $DATASET_ROOT: $(find "$DATASET_ROOT/data" -name '*.parquet' | wc -l | tr -d ' ') parquet," \
     "$(find "$DATASET_ROOT/videos" -name '*.mp4' | wc -l | tr -d ' ') mp4," \
     "$(du -sk "$DATASET_ROOT" | cut -f1) kB"
# The dataset is local by construction; a loader that cannot find it must
# fail here, not go looking for a Hub repository of the same name.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
