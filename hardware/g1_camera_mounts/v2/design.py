"""G1 camera mount revision 2. Millimetres, gripper frame +X toward tips, +Z up.
CadQuery 2.8; original geometry. Physical fit and load testing remain required.
"""
from pathlib import Path
import math,json
import cadquery as cq
import trimesh

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
DIMS=(57,60)
SLIDE=(-10,20)
TILT=(35,75)
NOMINAL_SLIDE=8
NOMINAL_TILT=60
PIVOT_X=45
PIVOT_Z=70
INNER_Y=40
OUTER_Y=46
FORK_INNER=46.3
FORK_OUTER=52.3
LOCK_R=25


def box(x0,x1,y0,y1,z0,z1):
 return cq.Workplane('XY').box(x1-x0,y1-y0,z1-z0).translate(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2))

def cyl_y(x,z,y0,y1,r):
 return cq.Workplane('XZ',origin=(x,y1,z)).circle(r).extrude(y1-y0)

def cyl_z(x,y,z0,z1,r):
 return cq.Workplane('XY',origin=(x,y,z0)).circle(r).extrude(z1-z0)

def deck_z(d):return (d+.6)/2+5+10

def clamp(d,upper=True):
 r=(d+.6)/2;R=r+5
 part=cq.Workplane('YZ',origin=(-9,0,0)).circle(R).circle(r).extrude(18)
 half=box(-60,60,-80,80,.75,100) if upper else box(-60,60,-80,80,-100,-.75)
 part=part.intersect(half)
 for side in [-1,1]:
  yc=side*(R+5)
  ear=box(-9,9,yc-7,yc+7,.75,9) if upper else box(-9,9,yc-7,yc+7,-9,-.75)
  part=part.union(ear.edges('|Z').fillet(2)).cut(cyl_z(0,yc,-12,12,2.25))
  if not upper:
   trap=cq.Workplane('XY',origin=(0,yc,-9.1)).polygon(6,7.25/math.cos(math.pi/6)).extrude(3.4)
   part=part.cut(trap)
 if upper:
  part=part.union(box(-9,9,-13,13,R-4,R+10))
  part=part.union(box(-9,50,-18,18,R+2,R+10).edges('|Z').fillet(2))
  for sy in [-1,1]:
   part=part.union(box(-9,50,sy*16.5-1.5,sy*16.5+1.5,R-2,R+3))
  for x in (17,41):
   for y in (-10,10):part=part.cut(cyl_z(x,y,R-3,R+12,2.25))
 return part.clean()


def arc_slot(x0,z0,y0,y1):
 # Exact circular arcs, rounded ends; x/z path = (-r cos a, r sin a).
 lo,hi=TILT;r=LOCK_R;w=1.7
 def p(rad,a):return (-rad*math.cos(math.radians(a)),rad*math.sin(math.radians(a)))
 a=p(r+w,lo);b=p(r+w,hi);c=p(r-w,hi);d=p(r-w,lo)
 plane=cq.Workplane('XZ',origin=(x0,y1,z0))
 cut=plane.moveTo(*a).threePointArc(p(r+w,(lo+hi)/2),b).lineTo(*c).threePointArc(p(r-w,(lo+hi)/2),d).close().extrude(y1-y0)
 for angle in (lo,hi):
  x,z=p(r,angle);cut=cut.union(cyl_y(x0+x,z0+z,y0,y1,w))
 return cut


def boom():
 # Four clamping screws in two longitudinal slots; 30 mm travel.
 p=box(-15,59,-20,20,0,6).edges('|Z').fillet(2)
 for y in (-10,10):p=p.cut(cq.Workplane('XY',origin=(24,y,-1)).slot2D(58.5,4.5,0).extrude(8))
 # Cross-beam and two broad fork cheeks; all above the gripper's moving envelope.
 p=p.union(box(32,58,-52.3,52.3,0,6))
 for side in (-1,1):
  y0,y1=(FORK_INNER,FORK_OUTER) if side==1 else (-FORK_OUTER,-FORK_INNER)
  outline=[(-11,-PIVOT_Z),(11,-PIVOT_Z),(11,-31),(31,-11),(31,11),(11,31),(-11,31),(-31,11),(-31,-11),(-11,-31)]
  cheek=cq.Workplane('XZ',origin=(PIVOT_X,y1,PIVOT_Z)).polyline(outline).close().extrude(y1-y0)
  cheek=cheek.cut(cyl_y(PIVOT_X,PIVOT_Z,y0-1,y1+1,2.25))
  cheek=cheek.cut(arc_slot(PIVOT_X,PIVOT_Z,y0-1,y1+1))
  p=p.union(cheek)
 # Two diagonal cross-beam braces from the narrow sliding spine to each cheek.
 for side in (-1,1):
  pts=[(side*15,6),(side*49,6),(side*49,28)]
  rib=cq.Workplane('YZ',origin=(37,0,0)).polyline(pts).close().extrude(16)
  p=p.union(rib)
 for y in (-10,10):p=p.cut(cq.Workplane('XY',origin=(24,y,-1)).slot2D(58.5,4.5,0).extrude(12))
 # Central access window for the tripod screwdriver at shallow tilt.
 p=p.cut(box(-12,31,-6,6,-1,8).edges('|Z').fillet(2))
 return p.clean()


def nut_trap(x,z,side,af,depth):
 # Open at the inner cheek face; ordinary hex nut is captive against rotation.
 if side==1:
  return cq.Workplane('XZ',origin=(x,INNER_Y+depth,z)).polygon(6,af/math.cos(math.pi/6)).extrude(depth+.1)
 return cq.Workplane('XZ',origin=(x,-INNER_Y+.1,z)).polygon(6,af/math.cos(math.pi/6)).extrude(depth+.1)


def moving_cheek(side):
 y0,y1=(40,46) if side==1 else (-46,-40)
 p=cyl_y(0,0,y0,y1,10).union(box(-25,0,y0,y1,-5,5)).union(cyl_y(-25,0,y0,y1,5))
 p=p.cut(cyl_y(0,0,y0-1,y1+1,2.25)).cut(cyl_y(-25,0,y0-1,y1+1,1.7))
 p=p.cut(nut_trap(0,0,side,7.25,3.4)).cut(nut_trap(-25,0,side,5.75,2.6))
 return p


def board_carrier():
 p=cq.Workplane('YZ').rect(40,40).extrude(4)
 p=p.cut(cq.Workplane('YZ',origin=(-1,0,0)).rect(20,20).extrude(6))
 for sy in (-1,1):
  for sz in (-1,1):p=p.cut(cq.Workplane('YZ',origin=(-1,sy*14,sz*14)).slot2D(2.4+math.sqrt(8),2.4,45*sy*sz).extrude(6))
 p=p.union(box(-6,1,-43,43,-6,6))
 # Split the bridge around an 8 mm rear cable/component opening.
 p=p.cut(box(-7,-.1,-8,8,-4,4))
 for side in (-1,1):
  p=p.union(moving_cheek(side))
  y0,y1=(40,46) if side==1 else (-46,-40)
  p=p.cut(cyl_y(0,0,y0-1,y1+1,2.25)).cut(cyl_y(-25,0,y0-1,y1+1,1.7))
  p=p.cut(nut_trap(0,0,side,7.25,3.4)).cut(nut_trap(-25,0,side,5.75,2.6))
 return p.clean()


def webcam_carrier():
 # Camera seating plane z=-24; optical centre is approximately level with hinge
 # when a webcam's lens is 24 mm above its tripod mounting plane.
 p=box(-30,30,-43,43,-30,-24).edges('|Z').fillet(2)
 p=p.cut(cq.Workplane('XY',origin=(0,0,-31)).slot2D(26.8,6.8,0).extrude(8))
 for y in (-34,34):p=p.cut(cq.Workplane('XY',origin=(0,y,-31)).slot2D(18,3,0).extrude(8))
 for side in (-1,1):
  y0,y1=(40,46) if side==1 else (-46,-40)
  wall=box(-27,18,y0,y1,-29,0)
  wall=wall.cut(box(-13,7,y0-1,y1+1,-22,-9))
  p=p.union(wall).union(moving_cheek(side))
  # Re-cut because wall intersects the cheek fastener holes and nut pockets.
  p=p.cut(cyl_y(0,0,y0-1,y1+1,2.25)).cut(cyl_y(-25,0,y0-1,y1+1,1.7))
  p=p.cut(nut_trap(0,0,side,7.25,3.4)).cut(nut_trap(-25,0,side,5.75,2.6))
 return p.clean()


def ring(od,id,h):return cq.Workplane('XY').circle(od/2).circle(id/2).extrude(h)

def gauge(d):
 r=(d+.6)/2
 return cq.Workplane('XY').circle(r+3).circle(r).extrude(2).intersect(box(-50,50,0,50,-1,3))

def parts():
 p={}
 for d in DIMS:
  p[f'clamp_upper_{d}mm']=(clamp(d), 'clamp')
  p[f'clamp_lower_{d}mm']=(clamp(d,False), 'clamp')
  p[f'fit_gauge_{d}mm']=(gauge(d),'flat')
 p['sliding_tilt_fork']=(boom(),'flat')
 p['so101_board_tilt_carrier']=(board_carrier(),'board')
 p['webcam_tilt_carrier']=(webcam_carrier(),'flat')
 p['m2_board_spacer_3mm']=(ring(5,2.4,3),'flat')
 p['m4_hinge_spacer_2mm']=(ring(10,4.5,2),'flat')
 p['m3_lock_spacer_2mm']=(ring(10,3.4,2),'flat')
 return p

def printable(p,orientation):
 if orientation=='clamp':p=p.rotate((0,0,0),(0,1,0),-90)
 if orientation=='board':p=p.rotate((0,0,0),(0,1,0),90)
 b=p.val().BoundingBox();return p.translate((0,0,-b.zmin))

def pose_carrier(p,d=60,slide=8,angle=60):
 return p.rotate((0,0,0),(0,1,0),angle).translate((-45+slide+PIVOT_X,0,deck_z(d)+PIVOT_Z))

def pose_boom(p,d=60,slide=8):return p.translate((-45+slide,0,deck_z(d)))

def mesh_shape(p):
 v,f=p.val().tessellate(.05,.08)
 m=trimesh.Trimesh(vertices=[v.toTuple() for v in v],faces=f,process=True)
 m.merge_vertices(digits_vertex=4);m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.remove_unreferenced_vertices()
 return m


def export_assemblies():
 for kind,carrier in [('board',board_carrier()),('webcam',webcam_carrier())]:
  a=cq.Assembly(name=f'g1_{kind}_revision_2')
  for name,part,col in [('upper',clamp(60).translate((-45,0,0)),(.08,.53,.54)),
                        ('lower',clamp(60,False).translate((-45,0,0)),(.08,.53,.54)),
                        ('fork',pose_boom(boom()),(.08,.53,.54)),
                        ('carrier',pose_carrier(carrier),(.91,.64,.24))]:
   a.add(part,name=name,color=cq.Color(*col))
  a.export(str(ROOT/'step'/f'assembly_{kind}_60mm.step'))


def main():
 (ROOT/'stl').mkdir(exist_ok=True);(ROOT/'step').mkdir(exist_ok=True)
 report={}
 for name,(part,orientation) in parts().items():
  assert part.val().isValid() and len(part.solids().vals())==1,name
  cq.exporters.export(part,str(ROOT/'step'/f'{name}.step'))
  p=printable(part,orientation)
  cq.exporters.export(p,str(ROOT/'stl'/f'{name}.stl'),tolerance=.025,angularTolerance=.08)
  m=trimesh.load(ROOT/'stl'/f'{name}.stl')
  m.merge_vertices(digits_vertex=4);m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.remove_unreferenced_vertices()
  m.apply_translation([0,0,-m.bounds[0,2]])
  m.export(ROOT/'stl'/f'{name}.stl');m=trimesh.load(ROOT/'stl'/f'{name}.stl')
  assert m.is_watertight and m.is_winding_consistent and m.volume>0,name
  assert abs(m.bounds[0,2])<.001,name
  report[name]={'watertight':True,'winding_consistent':True,'cad_solids':1,'volume_cm3':round(m.volume/1000,2),'print_size_mm':m.extents.round(2).tolist()}
  print(name,report[name],flush=True)
 (ROOT/'mesh_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 export_assemblies()

if __name__=='__main__':main()
