"""Reusable G1 mesh and segment-intersection utilities."""
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


def field_stats(origin,R,targets):
 local=(targets-origin)@R
 assert (local[:,0]>0).all()
 h=np.degrees(np.arctan2(local[:,1],local[:,0]));v=np.degrees(np.arctan2(local[:,2],local[:,0]))
 distances=np.linalg.norm(targets-origin,axis=1)
 return float(2*max(abs(h))),float(2*max(abs(v))),float(min(distances)),float(max(distances))


def robot_envelope():
 p=cq.Workplane('YZ',origin=(-76.66,0,0)).circle(30).extrude(62.66).union(box(-14,.01,-54.2,54.2,-30,30))
 for idx in (1,2):
  m0=robot_meshes(0)[idx];m1=robot_meshes(50)[idx]
  lo=np.minimum(m0.bounds[0],m1.bounds[0]);hi=np.maximum(m0.bounds[1],m1.bounds[1])
  p=p.union(box(lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]))
 return p


def overlap(a,b):return a.intersect(b).val().Volume()
