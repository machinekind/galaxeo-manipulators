# Sourced by the payloads that render (dataset generation, evaluation).
# Makes headless MuJoCo rendering work on a rented Linux box and sizes the
# CPU-bound work to the cores the box was actually sold with.
#
# EGL. A CUDA container image carries the driver's own libEGL_nvidia.so.0 but
# usually neither the GLVND front library PyOpenGL looks for (libEGL.so.1)
# nor the vendor file that tells GLVND where the driver's EGL is, and with
# MUJOCO_GL=egl `import mujoco` then dies in OpenGL/raw/EGL with
# "'NoneType' object has no attribute 'eglQueryString'". Both are a package
# and a four-line json away. Nothing happens when a render already works.
#
# Threads. Each worker process is one MuJoCo simulation, single threaded by
# nature, but numpy's OpenBLAS starts one thread per CPU it can see, and a
# container on a 88-core host shows all 88 whatever share it was sold: 40
# workers then want 3,500 threads and pthread_create fails. One BLAS thread
# per worker is the right number anyway.
#
# Cores. `nproc` reports the host; the cgroup quota is what the box may use.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cores_available() {
    local q p n all
    # not plain nproc: it answers OMP_NUM_THREADS when that is set, and this
    # file sets it to 1 (that once pinned a whole training run to one core)
    all=$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)
    n=""
    if [ -r /sys/fs/cgroup/cpu.max ]; then                       # cgroup v2
        read -r q p < /sys/fs/cgroup/cpu.max
        [ "$q" != max ] && [ "${p:-0}" -gt 0 ] 2>/dev/null && n=$(( q / p ))
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then       # cgroup v1
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$q" -gt 0 ] 2>/dev/null && [ "${p:-0}" -gt 0 ] 2>/dev/null && n=$(( q / p ))
    fi
    # a quota under two cores is not a box anyone rents for this; trust nproc then
    if [ -z "$n" ] || [ "$n" -lt 2 ] || [ "$n" -gt "$all" ]; then n=$all; fi
    echo "$n"
}

render_check() {
    python - <<'PY' 2>/dev/null
import mujoco
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size="1 1 .1" rgba=".5 .6 .7 1"/>'
    '<camera name="c" pos="0 -2 2" xyaxes="1 0 0 0 1 1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)          # without it the camera has no pose yet and the frame is black
r = mujoco.Renderer(m, 48, 64)
r.update_scene(d, camera="c")
img = r.render()
r.close()
raise SystemExit(0 if img.max() > 0 else 1)
PY
}

if [ "$(uname)" = Linux ]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    if ! render_check; then
        echo "headless render failed under MUJOCO_GL=$MUJOCO_GL; installing the GLVND EGL front"
        if command -v apt-get >/dev/null 2>&1; then
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libglvnd0 libopengl0 >/dev/null 2>&1 \
                || { DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 \
                     && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libglvnd0 libopengl0 >/dev/null 2>&1; } \
                || echo "WARN: apt-get could not install the EGL libraries" >&2
        fi
        if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] && ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 >/dev/null 2>&1; then
            mkdir -p /usr/share/glvnd/egl_vendor.d
            printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
                > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
        fi
        if render_check; then
            echo "headless render ok after the EGL setup"
        else
            echo "ERROR: MuJoCo cannot render headless on this box (MUJOCO_GL=$MUJOCO_GL)" >&2
            exit 1
        fi
    fi
fi
