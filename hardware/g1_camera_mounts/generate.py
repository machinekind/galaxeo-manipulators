"""Original parametric G1 camera mounts, millimetres. Requires cadquery + trimesh.

Run: python generate.py
STLs are oriented for printing; STEP files retain assembly coordinates.
Prototype fit must be checked on the physical gripper. No rated load is implied.
"""
from pathlib import Path
import json
import math
import cadquery as cq
import trimesh

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stl"
STEP = ROOT / "step"
DIAMETERS = (57.0, 60.0)
BORE_ALLOWANCE = 0.6  # diametral allowance, with nominal 0.5 mm rubber liner
WALL = 5.0
WIDTH = 18.0
GAP = 1.5
CAMERA_TILT = 20.0  # optical axis down from the gripper's forward axis


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1-x0, y1-y0, z1-z0).translate(
        ((x0+x1)/2, (y0+y1)/2, (z0+z1)/2))


def zhole(x, y, radius, z0, height):
    return cq.Workplane("XY", origin=(x,y,z0)).circle(radius).extrude(height)


def deck_height(d):
    return (d+BORE_ALLOWANCE)/2 + WALL + 6


def clamp(d, upper=True):
    r = (d+BORE_ALLOWANCE)/2
    outer = r+WALL
    ring = cq.Workplane("YZ", origin=(-WIDTH/2,0,0)).circle(outer).circle(r).extrude(WIDTH)
    half = box(-50,50,-70,70,GAP/2,80) if upper else box(-50,50,-70,70,-80,-GAP/2)
    part = ring.intersect(half)
    for side in (-1,1):
        yc = side*(outer+5)
        ear = box(-9,9,yc-7,yc+7,GAP/2,9) if upper else box(-9,9,yc-7,yc+7,-9,-GAP/2)
        ear = ear.edges("|Z").fillet(2)
        part = part.union(ear).cut(zhole(0,yc,2.25,-12,24))
        if not upper:
            # Nut captured from below; hex flats 7.25 mm for an M4 nut.
            nut = cq.Workplane("XY",origin=(0,yc,-9.1)).polygon(6,7.25/math.cos(math.pi/6)).extrude(3.4)
            part = part.cut(nut)
    if upper:
        pedestal = box(-9,9,-13,13,outer-4,outer+6)
        deck = box(-9,25,-16,16,outer+1,outer+6).edges("|Z").fillet(2)
        part = part.union(pedestal).union(deck)
        for y in (-10,10):
            part = part.cut(zhole(17,y,1.7,outer-2,12))
    return part.clean()


def adapter_base(thickness=4, end=30, half_width=16):
    part = box(9,end,-half_width,half_width,0,thickness).edges("|Z").fillet(2)
    for y in (-10,10):
        part = part.cut(zhole(17,y,1.7,-1,thickness+2))
    return part


def board_adapter():
    # Board attaches to forward face using four M2 bolts + insulating spacers.
    # Camera-hole square pitch continuously adjustable 26..30 mm via diagonal slots.
    frame = cq.Workplane("YZ").rect(40,40).extrude(4)
    frame = frame.cut(cq.Workplane("YZ",origin=(-1,0,0)).rect(20,20).extrude(6))
    for sy in (-1,1):
        for sz in (-1,1):
            plane = cq.Workplane("YZ",origin=(-1,sy*14,sz*14))
            frame = frame.cut(plane.slot2D(2.4+math.sqrt(8),2.4,45*sy*sz).extrude(6))
    # Lower edge of frame sits in the base; lens looks forwards and 20 degrees down.
    frame = frame.translate((0,0,20)).rotate((0,0,0),(0,1,0),CAMERA_TILT).translate((24,0,2))
    part = adapter_base(half_width=20).union(frame)
    # Side braces leave the PCB fasteners and the two base screws accessible.
    for y in (-18,18):
        rib = cq.Workplane("XZ",origin=(0,y+1.5,0)).polyline(
            [(12,2),(32,2),(32,24)]).close().extrude(3)
        part = part.union(rib)
    # Horizontal cable-tie eye in the extended rear edge of the base.
    part = part.cut(cq.Workplane("XY",origin=(12,0,-1)).slot2D(9,2.5,90).extrude(6))
    return part.clean()


def webcam_adapter():
    part = adapter_base(thickness=6,end=61,half_width=24)
    for y in (-10,10):
        # Recess M3 socket heads below the webcam's seating surface.
        part = part.cut(zhole(17,y,3.2,2.8,4))
    # Slot's screw-centre travel is x=35..49 mm; use a metal 1/4-20 screw + washer.
    slot = cq.Workplane("XY",origin=(42,0,-1)).slot2D(20.8,6.8,0).extrude(8)
    part = part.cut(slot)
    for y in (-19,19):
        part = part.cut(cq.Workplane("XY",origin=(48,y,-1)).slot2D(12,3,0).extrude(8))
    return part.clean()


def fit_gauge(d):
    # Thin open semicircle checks OD/access; not a load-carrying clamp.
    r=(d+BORE_ALLOWANCE)/2
    ring=cq.Workplane("XY").circle(r+3).circle(r).extrude(2)
    return ring.intersect(box(-50,50,0,50,-1,3)).clean()


def oriented(part, collar=False):
    p=part.rotate((0,0,0),(0,1,0),-90) if collar else part
    bb=p.val().BoundingBox()
    return p.translate((0,0,-bb.zmin))


def make_parts():
    parts={}
    for d in DIAMETERS:
        suffix=f"{d:g}mm"
        parts[f"clamp_upper_{suffix}"]=(clamp(d,True),True)
        parts[f"clamp_lower_{suffix}"]=(clamp(d,False),True)
        parts[f"fit_gauge_{suffix}"]=(fit_gauge(d),False)
    parts["so101_32mm_camera_adapter"]=(board_adapter(),False)
    parts["tripod_webcam_adapter"]=(webcam_adapter(),False)
    parts["m2_camera_spacer_3mm"]=(cq.Workplane("XY").circle(2.5).circle(1.2).extrude(3),False)
    return parts


def main():
    OUT.mkdir(exist_ok=True); STEP.mkdir(exist_ok=True)
    report={}
    for name,(part,is_collar) in make_parts().items():
        assert part.val().isValid(), f"Invalid CAD: {name}"
        assert len(part.solids().vals())==1, f"Disconnected CAD: {name}"
        print_part=oriented(part,is_collar)
        cq.exporters.export(print_part,str(OUT/f"{name}.stl"),tolerance=0.03,angularTolerance=0.08)
        cq.exporters.export(part,str(STEP/f"{name}.step"))
        m=trimesh.load(OUT/f"{name}.stl",force="mesh")
        # OCC can duplicate seam vertices by tens of nanometres at fillet joins.
        # Weld at 0.0001 mm (well below tessellation/printing tolerance).
        m.merge_vertices(digits_vertex=4)
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        m.export(OUT/f"{name}.stl")
        m=trimesh.load(OUT/f"{name}.stl",force="mesh")
        assert m.is_watertight and m.is_winding_consistent and m.volume>0, name
        assert abs(m.bounds[0,2])<0.001, f"Not on bed: {name}"
        report[name]={"watertight":bool(m.is_watertight),"consistent_winding":bool(m.is_winding_consistent),
            "cad_solids":len(part.solids().vals()),"triangles":len(m.faces),
            "print_dimensions_mm":[round(float(v),2) for v in m.extents],
            "volume_cm3":round(float(m.volume/1000),2)}
        print(name,report[name],flush=True)
    # Check adapter-to-deck registration and no unintended solid intersection.
    for d in DIAMETERS:
        upper=clamp(d,True);lower=clamp(d,False)
        assert upper.intersect(lower).val().Volume()<1e-6
        for adapter in (board_adapter(),webcam_adapter()):
            assert upper.intersect(adapter.translate((0,0,deck_height(d)))).val().Volume()<1e-6
    # Conservative static collision envelopes from the repository's G1 mesh.
    # Camera housings, wires, arm links and full robot motion are not covered.
    repo=ROOT.parents[1]
    meshes=repo/"ros2_ws/src/galaxea_a1xy_description/meshes"
    if not (meshes/"gripper_link.STL").exists():
        report["assembly_checks"]={"adapter_deck_overlap_mm3":0,"clamp_half_overlap_mm3":0,
            "g1_envelope_check":"Skipped: repository gripper meshes unavailable in this standalone copy."}
        (ROOT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
        return
    housing=cq.Workplane("YZ",origin=(-76.66,0,0)).circle(30).extrude(62.66)
    rail=box(-14,0.01,-54.2,54.2,-30,30)
    robot=housing.union(rail)
    for idx,sign in ((1,1),(2,-1)):
        m=trimesh.load(meshes/f"gripper_finger_link{idx}.STL")
        bounds=m.bounds*1000
        bounds += [36.89,sign*13.453,sign*0.12059]
        # Entire linear jaw travel, 0..50 mm each side.
        if sign==1: bounds[1,1]+=50
        else: bounds[0,1]-=50
        robot=robot.union(box(bounds[0,0],bounds[1,0],bounds[0,1],bounds[1,1],bounds[0,2],bounds[1,2]))
    for name,part in (("upper",clamp(60,True)),("lower",clamp(60,False)),
                      ("board",board_adapter().translate((0,0,deck_height(60)))),
                      ("webcam",webcam_adapter().translate((0,0,deck_height(60))))):
        overlap=part.translate((-45,0,0)).intersect(robot).val().Volume()
        assert overlap<1e-6, f"G1 envelope collision: {name} {overlap}"
    report["assembly_checks"]={"adapter_deck_overlap_mm3":0,"clamp_half_overlap_mm3":0,
        "g1_60mm_static_envelope_overlap_mm3":0,"clamp_center_gripper_frame_x_mm":-45,
        "jaw_travel_checked_mm_per_side":50,
        "limitations":"No physical fit, load, camera-body, cable, or full-arm motion validation."}
    (ROOT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
