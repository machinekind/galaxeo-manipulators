"""galaxeo: host-side tools for the Galaxea A1X arm over CAN-FD.

    from galaxeo import protocol, bus

protocol - CAN ids, field scales, encoders and the feedback decoder (stdlib, pure).
bus      - transports behind one recv/send/close interface: SocketCAN (stdlib),
           the XCAN dongle on macOS (extra "xcan"), python-can (extra "pcan").
xcan_usb - the userspace libusb driver behind the XCAN transport.

Importing any of them needs only the standard library.

arm      - the installed arm: model, IK, collisions, reach box, guarded moves
           (extra "arm": numpy, mujoco).
robot    - the A1X as a robot package for the digital twin: robot folder and the
           `a1x` backend (standard library only; the twin finds it by entry points).
grasp    - the wrist-camera grasp student as a strategy for the digital twin
           (extra "grasp": numpy, torch, opencv-contrib-python).
"""

__version__ = "0.1.0"
