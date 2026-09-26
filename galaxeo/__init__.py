"""galaxeo: host-side tools for the Galaxea A1X arm over CAN-FD.

    from galaxeo import protocol, bus

protocol - CAN ids, field scales, encoders and the feedback decoder (stdlib, pure).
bus      - transports behind one recv/send/close interface: SocketCAN (stdlib),
           the XCAN dongle on macOS (extra "xcan"), python-can (extra "pcan").
xcan_usb - the userspace libusb driver behind the XCAN transport.

Importing any of them needs only the standard library.

arm      - the installed arm: model, IK, collisions, reach box, guarded moves
           (extra "arm": numpy, mujoco).
"""

__version__ = "0.1.0"
