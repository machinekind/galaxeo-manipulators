"""Mesh-based segment visibility + solid mechanical checks for revision 2."""
import json,math
import numpy as np
import trimesh
import cadquery as cq
from design import *


def robot_meshes(q):
 folder=REPO/'ros2_ws/src/galaxea_a1xy_description/meshes'
 out=[]
 for name,offset in [('gripper_link.STL',[0,0,0]),('gripper_finger_link1.STL',[36.89,13.453+q,.12059]),('gripper_finger_link2.STL',[36.89,-13.453-q,-.12059])]:
  m=trimesh.load(folder/name);m.apply_scale(1000);m.apply_translation(offset);out.append(m)
 return out


def blocked(origin,targets,triangles):
 # Moller-Trumbore line-segment intersections, two-sided triangles.
 a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
 tvec=origin-a;qvec=np.cross(tvec,e1);dot2=np.einsum('ij,ij->i',e2,qvec)
 out=[]
 for target in targets:
  ray=target-origin;pvec=np.cross(np.broadcast_to(ray,e2.shape),e2)
  det=np.einsum('ij,ij->i',e1,pvec)
  inv=np.divide(1.,det,out=np.zeros_like(det),where=abs(det)>1e-10)
  u=np.einsum('ij,ij->i',tvec,pvec)*inv
  v=np.einsum('j,ij->i',ray,qvec)*inv;t=dot2*inv
  hit=(abs(det)>1e-10)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>1e-5)&(t<1-1e-5)
  out.append(bool(hit.any()))
 return np.array(out)


def rotation(angle):
 t=math.radians(angle);c,s=math.cos(t),math.sin(t)
 return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def camera_pose(kind='board',d=60,slide=8,height=24,forward=20):
 z=deck_z(d)+PIVOT_Z;dx=75-slide
 lz=0 if kind=='board' else height-24
 angle=math.degrees(math.atan2(z,dx)+math.asin(lz/math.hypot(dx,z)))
 local=np.array([14.6,0,0]) if kind=='board' else np.array([forward,0,lz])
 R=rotation(angle);origin=np.array([slide,0,z])+R@local
 return origin,R,angle


def field_stats(origin,R,targets):
 local=(targets-origin)@R
 assert (local[:,0]>0).all()
 h=np.degrees(np.arctan2(local[:,1],local[:,0]));v=np.degrees(np.arctan2(local[:,2],local[:,0]))
 distances=np.linalg.norm(targets-origin,axis=1)
 return float(2*max(abs(h))),float(2*max(abs(v))),float(min(distances)),float(max(distances))


def optics():
 report=[]
 fixed={d:np.concatenate([mesh_shape(clamp(d).translate((-45,0,0))).triangles,mesh_shape(clamp(d,False).translate((-45,0,0))).triangles]) for d in DIMS}
 fork_tri=mesh_shape(boom()).triangles
 carrier_tri={'board':mesh_shape(board_carrier()).triangles,'webcam':mesh_shape(webcam_carrier()).triangles}
 robot_tri={q:np.concatenate([m.triangles for m in robot_meshes(q)]) for q in (5,10,20,35,50)}
 for kind in ('board','webcam'):
  cases=[(24,14.6)] if kind=='board' else [(h,x) for h in (10,24,40) for x in (0,20,32.5)]
  for d in DIMS:
   for slide in SLIDE:
    for height,forward in cases:
     origin,R,angle=camera_pose(kind,d,slide,height,forward)
     assert TILT[0]<=angle<=TILT[1]
     fixture_tri=np.concatenate([fixed[d],fork_tri+[-45+slide,0,deck_z(d)],carrier_tri[kind]@R.T+[slide,0,deck_z(d)+PIVOT_Z]])
     maxh=maxv=0;near=1e9;far=0;count=0;occluded=0
     for q in (5,10,20,35,50):
      # Visible empty grasp region, not surfaces physically hidden inside fingers.
      targets=np.array([[x,y,0] for x in np.linspace(60,76,5) for y in np.linspace(-q+2,q-2,9)])
      tri=np.concatenate([robot_tri[q],fixture_tri])
      mask=blocked(origin,targets,tri);occluded+=int(mask.sum());count+=len(targets)
      h,v,n,f=field_stats(origin,R,targets);maxh=max(maxh,h);maxv=max(maxv,v);near=min(near,n);far=max(far,f)
     assert occluded==0,(kind,d,slide,height,forward,occluded)
     # Aim test is independent of focal length or selected FoV.
     axis=R[:,0];ground=origin+axis*(-origin[2]/axis[2])
     assert np.linalg.norm(ground-[75,0,0])<1e-8
     report.append(dict(kind=kind,diameter_mm=d,slide_mm=slide,lens_height_over_webcam_base_mm=height if kind=='webcam' else None,
       lens_forward_offset_mm=forward if kind=='webcam' else 14.6,angle_deg=round(angle,2),lens_xyz_mm=origin.round(2).tolist(),
       rays=count,occluded=occluded,min_horizontal_fov_deg=round(maxh,2),min_vertical_fov_deg=round(maxv,2),
       target_distance_range_mm=[round(near,2),round(far,2)]))
     print('OPTICS',kind,d,slide,height,forward,'angle',round(angle,1),'occluded',occluded,'HFOV',round(maxh,1),flush=True)
 (ROOT/'sightline_validation.json').write_text(json.dumps({'method':'Two-sided segment/triangle intersection against G1 body, both fingers and all four printed assembly parts.',
  'target_region':'X=60..76 mm, Z=0, Y=(-q+2)..(q-2), q=5/10/20/35/50 mm per jaw; 45 targets per opening.',
  'camera_assumptions':'Board optical centre 14.6 mm ahead of hinge. Webcam optical centre on lateral centreline, 10..40 mm above mounting plane and 0..32.5 mm forward of hinge in camera coordinates.',
  'limits':'Ray samples are geometric, not a physical camera calibration. No promise about lens distortion, focus, scene lighting, object self-occlusion or unmodeled cabling. A closed gripper has no visible empty gap.',
  'cases':report},indent=2)+'\n')
 return report


def robot_envelope():
 p=cq.Workplane('YZ',origin=(-76.66,0,0)).circle(30).extrude(62.66).union(box(-14,.01,-54.2,54.2,-30,30))
 for idx in (1,2):
  m0=robot_meshes(0)[idx];m1=robot_meshes(50)[idx]
  lo=np.minimum(m0.bounds[0],m1.bounds[0]);hi=np.maximum(m0.bounds[1],m1.bounds[1])
  p=p.union(box(lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]))
 return p


def overlap(a,b):return a.intersect(b).val().Volume()


def mechanics():
 report=[];robot=robot_envelope();fork=boom();board=board_carrier();webcam=webcam_carrier()
 for d in DIMS:
  upper=clamp(d).translate((-45,0,0));lower=clamp(d,False).translate((-45,0,0))
  assert overlap(upper,lower)<1e-6
  if d==60:
   assert overlap(upper,robot)<1e-6;assert overlap(lower,robot)<1e-6
  for slide in (-10,5,20):
   f=pose_boom(fork,d,slide)
   assert overlap(f,upper)<1e-6
   if d==60:assert overlap(f,robot)<1e-6
   for angle in range(35,76,5):
    for kind,carrier in [('board',board),('webcam',webcam)]:
     c=pose_carrier(carrier,d,slide,angle)
     hits={'fork':overlap(c,f),'upper':overlap(c,upper)}
     if d==60:hits['robot_swept_envelope']=overlap(c,robot)
     assert max(hits.values())<1e-5,(d,slide,angle,kind,hits)
     report.append([d,slide,angle,kind])
   print('MECHANICS',d,slide,'passed',flush=True)
 # Validate a stated compact-webcam envelope (not every arbitrary webcam).
 for slide in (-10,20):
  f=pose_boom(fork,60,slide)
  for angle in (35,45,55,65,75):
   for socket_shift in (-10,10):
    camera=box(socket_shift-22.5,socket_shift+22.5,-37,37,-24,21)
    camera=pose_carrier(camera,60,slide,angle)
    assert overlap(camera,f)<1e-6
    assert overlap(camera,robot)<1e-6
 # Fastener axes: four M4 boom screws fit both slots throughout slide range;
 # M3 lock axes pass through curved slots across entire tilt range.
 for slide in (-10,5,20):
  f=pose_boom(fork,60,slide)
  for x in (17,41):
   for y in (-10,10):
    screw=cyl_z(x-45,y,deck_z(60)-8,deck_z(60)+6,2)
    assert overlap(screw,f)<1e-6
 for angle in range(35,76):
  x=-25*math.cos(math.radians(angle));z=25*math.sin(math.radians(angle))
  for s in (-1,1):
   lock=cyl_y(PIVOT_X+x,PIVOT_Z+z,s*49.3-3,s*49.3+3,1.5)
   assert overlap(lock,fork)<1e-6,(angle,s)
 (ROOT/'mechanical_validation.json').write_text(json.dumps({'collision_samples':len(report),'samples':report,
  'results':'No solid overlap at checked configurations; 60 mm set clears conservative housing and full-jaw-sweep envelopes.',
  'webcam_body_envelope_mm':[45,74,45],'webcam_socket_shift_checked_mm':[-10,10],
  'limits':'57 mm set checked internally only; physical housing not available. No load, vibration, creep, heat or full-arm workspace testing.'},indent=2)+'\n')

if __name__=='__main__':
 mechanics();optics()
