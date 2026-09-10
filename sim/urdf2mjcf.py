#!/usr/bin/env python3
"""Generate sim/a1x.xml (MuJoCo MJCF) from the A1X URDF in the description package.

    sim/.venv/bin/python sim/urdf2mjcf.py

Keeps the URDF's masses, inertias, joint origins, axes, limits and effort
ratings. Adds what a URDF cannot express: position actuators per joint,
a mimic-style coupling between the two gripper fingers, visual/collision
geom classes and a "home" keyframe. Rerun after editing the URDF.
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "ros2_ws", "src", "galaxea_a1xy_description")
URDF = os.path.join(PKG, "urdf", "a1x.urdf")
OUT = os.path.join(HERE, "a1x.xml")
MESHDIR = os.path.relpath(os.path.join(PKG, "meshes"), HERE)

# Position-servo gains. Roughly "stiff enough to hold pose under gravity".
KP = {"arm_joint1": 300, "arm_joint2": 600, "arm_joint3": 400,
      "arm_joint4": 150, "arm_joint5": 100, "arm_joint6": 60}
KP_GRIPPER = 400
HOME = {"arm_joint2": 1.0, "arm_joint3": -1.6, "arm_joint4": 0.6,
        "gripper_finger_joint1": 0.03}


def fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def main():
    robot = ET.parse(URDF).getroot()
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = robot.findall("joint")
    children = {}          # parent link -> [joint]
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
    root_link = next(n for n in links if n not in
                     {j.find("child").get("link") for j in joints})

    mj = ET.Element("mujoco", model=robot.get("name"))
    ET.SubElement(mj, "compiler", angle="radian", eulerseq="XYZ",
                  meshdir=MESHDIR, autolimits="true")
    ET.SubElement(mj, "option", integrator="implicitfast")

    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", armature="0.01", damping="1")
    ET.SubElement(default, "position", forcelimited="true")
    d_vis = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(d_vis, "geom", type="mesh", contype="0", conaffinity="0",
                  group="2")
    d_col = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(d_col, "geom", type="mesh", group="3")

    asset = ET.SubElement(mj, "asset")
    meshes = set()
    world = ET.SubElement(mj, "worldbody")
    actuators = ET.SubElement(mj, "actuator")
    equality = ET.SubElement(mj, "equality")
    joint_order = []

    def add_link(parent_el, link_name, pos="0 0 0", euler="0 0 0", joint=None):
        link = links[link_name]
        body = ET.SubElement(parent_el, "body", name=link_name, pos=pos)
        if euler != "0 0 0":
            body.set("euler", euler)

        if joint is not None:
            jtype = joint.get("type")
            lim = joint.find("limit")
            j = ET.SubElement(body, "joint", name=joint.get("name"),
                              type="hinge" if jtype == "revolute" else "slide",
                              axis=joint.find("axis").get("xyz"))
            if lim is not None:
                j.set("range", f'{lim.get("lower")} {lim.get("upper")}')
                joint_order.append((joint.get("name"), jtype,
                                    float(lim.get("lower")), float(lim.get("upper")),
                                    float(lim.get("effort"))))

        inr = link.find("inertial")
        if inr is not None:
            o = inr.find("origin")
            assert o is None or o.get("rpy", "0 0 0").split() == ["0", "0", "0"], link_name
            i = inr.find("inertia")
            ET.SubElement(body, "inertial",
                          pos=o.get("xyz") if o is not None else "0 0 0",
                          mass=inr.find("mass").get("value"),
                          fullinertia=" ".join(i.get(k) for k in
                                               ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))

        for tag, cls in (("visual", "visual"), ("collision", "collision")):
            for el in link.findall(tag):
                mesh = el.find("geometry/mesh")
                if mesh is None:
                    continue
                fname = os.path.basename(mesh.get("filename"))
                mname = os.path.splitext(fname)[0]
                if fname not in meshes:
                    meshes.add(fname)
                    ET.SubElement(asset, "mesh", name=mname, file=fname)
                g = ET.SubElement(body, "geom", {"class": cls}, mesh=mname)
                o = el.find("origin")
                if o is not None and o.get("xyz", "0 0 0") != "0 0 0":
                    g.set("pos", o.get("xyz"))
                if o is not None and o.get("rpy", "0 0 0") != "0 0 0":
                    g.set("euler", o.get("rpy"))
                color = el.find("material/color")
                if cls == "visual" and color is not None:
                    g.set("rgba", color.get("rgba"))

        for cj in children.get(link_name, []):
            o = cj.find("origin")
            add_link(body, cj.find("child").get("link"),
                     pos=o.get("xyz", "0 0 0") if o is not None else "0 0 0",
                     euler=o.get("rpy", "0 0 0") if o is not None else "0 0 0",
                     joint=None if cj.get("type") == "fixed" else cj)

    add_link(world, root_link)

    # Actuators: one position servo per arm joint, one for the gripper.
    for name, jtype, lo, hi, effort in joint_order:
        if name == "gripper_finger_joint2":
            continue                        # driven through the equality below
        kp = KP_GRIPPER if name == "gripper_finger_joint1" else KP[name]
        ET.SubElement(actuators, "position", name=name.replace("_joint", ""),
                      joint=name, kp=str(kp), kv=str(kp / 20),
                      ctrlrange=fmt((lo, hi)), forcerange=fmt((-effort, effort)))
    ET.SubElement(equality, "joint", joint1="gripper_finger_joint2",
                  joint2="gripper_finger_joint1", polycoef="0 -1 0 0 0")

    # Keyframe "home": a raised pose so the arm doesn't rest on its own base.
    qpos = [HOME.get(n, 0.0) for n, *_ in joint_order]
    qpos[joint_order.index(next(j for j in joint_order if j[0] == "gripper_finger_joint2"))] = \
        -HOME["gripper_finger_joint1"]
    ctrl = [HOME.get(n, 0.0) for n, *_ in joint_order if n != "gripper_finger_joint2"]
    kf = ET.SubElement(mj, "keyframe")
    ET.SubElement(kf, "key", name="home", qpos=fmt(qpos), ctrl=fmt(ctrl))

    xml = minidom.parseString(ET.tostring(mj)).toprettyxml(indent="  ")
    xml = xml.replace('<?xml version="1.0" ?>\n', "")
    with open(OUT, "w") as f:
        f.write(f"<!-- Generated by sim/urdf2mjcf.py from {os.path.relpath(URDF, HERE)}. "
                "Do not edit by hand. -->\n")
        f.write(xml)
    print("wrote", os.path.relpath(OUT), "with", len(joint_order), "joints,",
          len(meshes), "meshes")


if __name__ == "__main__":
    main()
