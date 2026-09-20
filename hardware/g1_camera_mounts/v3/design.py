"""Fixed 60-degree G1 camera brackets, revision 3. All dimensions in mm.
Print one integrated upper bracket and one lower clamp. No adjustment joints.
"""
from pathlib import Path
import math,json
import cadquery as cq
import trimesh
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
ANGLE=60
DIMS=(57,60)

def box(x0,x1,y0,y1,z0,z1):
 return cq.Workplane('XY').box(x1-x0,y1-y0,z1-z0).translate(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2))

def cylinder_z(x,y,z0,z1,r):return cq.Workplane('XY',origin=(x,y,z0)).circle(r).extrude(z1-z0)

def rotation():
 a=math.radians(ANGLE);c,s=math.cos(a),math.sin(a)
 return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def reference(d):return np.array([13.5,0,106.5+(d-60)/2])

def camera_plane(kind,d):
 # The webcam seating plane is 24 mm below the board-camera reference axis.
 return reference(d) if kind=='board' else reference(d)+rotation()@np.array([0,0,-24])

def pose(p,kind,d):return p.rotate((0,0,0),(0,1,0),ANGLE).translate(tuple(camera_plane(kind,d)))

def clamp(d,upper):
 r=(d+.6)/2;R=r+5
 p=cq.Workplane('YZ',origin=(-54,0,0)).circle(R).circle(r).extrude(18)
 half=box(-60,-30,-70,70,.75,60) if upper else box(-60,-30,-70,70,-60,-.75)
 p=p.intersect(half)
 for side in (-1,1):
  y=side*(R+5)
  ear=box(-54,-36,y-7,y+7,.75,9) if upper else box(-54,-36,y-7,y+7,-9,-.75)
  p=p.union(ear.edges('|Z').fillet(2)).cut(cylinder_z(-45,y,-12,12,2.25))
  if not upper:
   pocket=cq.Workplane('XY',origin=(-45,y,-9.1)).polygon(6,7.25/math.cos(math.pi/6)).extrude(3.4)
   p=p.cut(pocket)
 return p.clean()

def board_frame():
 p=cq.Workplane('YZ').rect(40,40).extrude(4)
 p=p.cut(cq.Workplane('YZ',origin=(-1,0,0)).rect(20,20).extrude(6))
 for sy in (-1,1):
  for sz in (-1,1):p=p.cut(cq.Workplane('YZ',origin=(-1,sy*14,sz*14)).slot2D(2.4+math.sqrt(8),2.4,45*sy*sz).extrude(6))
 # Small side tabs join the fixed ribs outside the PCB screw/nut envelope.
 for side in (-1,1):p=p.union(box(-1,4,side*20.5-2.5,side*20.5+2.5,-20,-12))
 return p

def webcam_plate():
 p=box(-25,25,-27,27,-6,0).edges('|Z').fillet(2)
 p=p.cut(cylinder_z(0,0,-7,1,3.4))
 return p

def upper(kind,d):
 p=clamp(d,True);R=rotation();C=camera_plane(kind,d)
 plate=board_frame() if kind=='board' else webcam_plate()
 p=p.union(pose(plate,kind,d))
 if kind=='board':
  ends=[C+R@np.array([5,0,-14]),C+R@np.array([-3,0,-14])]
  ymin,ymax=18,22;root_z=27+(d-60)/2
 else:
  ends=[C+R@np.array([20,0,-5.5]),C+R@np.array([-20,0,-5.5])]
  ymin,ymax=20,24;root_z=25+(d-60)/2
 profile=[(-54,root_z),(-36,root_z),(ends[0][0],ends[0][2]),(ends[1][0],ends[1][2])]
 for side in (-1,1):
  y0,y1=(ymin,ymax) if side==1 else (-ymax,-ymin)
  rib=cq.Workplane('XZ',origin=(0,y1,0)).polyline(profile).close().extrude(y1-y0)
  p=p.union(rib)
 return p.clean()

def gauge(d):
 r=(d+.6)/2
 return cq.Workplane('XY').circle(r+3).circle(r).extrude(2).intersect(box(-50,50,0,50,-1,3))

def parts():
 out={}
 for d in DIMS:
  for kind in ('board','webcam'):out[f'{kind}_fixed60_upper_{d}mm']=(upper(kind,d),'side')
  out[f'lower_clamp_{d}mm']=(clamp(d,False),'axial')
  out[f'fit_gauge_{d}mm']=(gauge(d),'flat')
 out['m2_spacer_3mm']=(cq.Workplane('XY').circle(2.5).circle(1.2).extrude(3),'flat')
 return out

def mesh_shape(p):
 v,f=p.val().tessellate(.04,.08)
 m=trimesh.Trimesh(vertices=[v.toTuple() for v in v],faces=f,process=True)
 m.merge_vertices(digits_vertex=4);m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.remove_unreferenced_vertices()
 return m

def print_pose(p,orientation):
 if orientation=='axial':p=p.rotate((0,0,0),(0,1,0),-90)
 # Sideways printing gives continuous layers along the fixed supporting ribs.
 if orientation=='side':p=p.rotate((0,0,0),(1,0,0),90)
 b=p.val().BoundingBox();return p.translate((0,0,-b.zmin))

def main():
 (ROOT/'stl').mkdir(exist_ok=True);(ROOT/'step').mkdir(exist_ok=True)
 report={}
 for name,(p,orientation) in parts().items():
  assert p.val().isValid() and len(p.solids().vals())==1,name
  cq.exporters.export(p,str(ROOT/'step'/f'{name}.step'))
  cq.exporters.export(print_pose(p,orientation),str(ROOT/'stl'/f'{name}.stl'),tolerance=.025,angularTolerance=.08)
  m=trimesh.load(ROOT/'stl'/f'{name}.stl');m.merge_vertices(digits_vertex=2);m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.remove_unreferenced_vertices()
  m.apply_translation([0,0,-m.bounds[0,2]]);m.export(ROOT/'stl'/f'{name}.stl')
  m=trimesh.load(ROOT/'stl'/f'{name}.stl')
  assert m.is_watertight and m.is_winding_consistent and m.volume>0,name
  report[name]={'watertight':True,'single_cad_solid':True,'volume_cm3':round(float(m.volume/1000),2),'print_dimensions_mm':m.extents.round(2).tolist()}
  print(name,report[name],flush=True)
 (ROOT/'mesh_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 for kind in ('board','webcam'):
  a=cq.Assembly(name=f'fixed_{kind}_60mm');a.add(upper(kind,60),name='upper',color=cq.Color(.12,.55,.55));a.add(clamp(60,False),name='lower',color=cq.Color(.12,.55,.55));a.export(str(ROOT/'step'/f'assembly_{kind}_60mm.step'))

if __name__=='__main__':main()
