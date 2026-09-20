"""Verify fixed-angle parts, fasteners, camera envelopes and sightlines."""
from design import *
from geometry_checks import robot_meshes,blocked,field_stats,robot_envelope,overlap

def main():
 robot=robot_envelope();cases=[];mechanical=0;R=rotation()
 for d in DIMS:
  lower=clamp(d,False)
  for side in (-1,1):
   y=side*((d+.6)/2+10)
   nut=cq.Workplane('XY',origin=(CLAMP_X,y,-8.9)).polygon(6,7/math.cos(math.pi/6)).extrude(3.2)
   assert overlap(nut,lower)<1e-6
  for kind in ('board','webcam'):
   top=upper(kind,d)
   assert overlap(top,lower)<1e-6
   # Shaft, washer and head of the two M4x20 collar screws.
   for side in (-1,1):
    y=side*((d+.6)/2+10)
    bolt=cylinder_z(CLAMP_X,y,-10.2,9.8,2).union(cylinder_z(CLAMP_X,y,9,9.8,4.5)).union(cylinder_z(CLAMP_X,y,9.8,13.8,3.5))
    assert overlap(bolt,top)<1e-6
   if d==60:
    assert overlap(top,robot)<1e-6,(kind,'robot')
    assert overlap(lower,robot)<1e-6
   if kind=='webcam':
    # Compact webcam seated on z=0 of the fixed plate. No side walls.
    camera=pose(box(-22.5,22.5,-37,37,0,45),kind,d)
    assert overlap(camera,top)<1e-6
    driver=pose(cylinder_z(0,0,-57,-6.1,5),kind,d)
    washer=pose(cylinder_z(0,0,-7,-6,6).cut(cylinder_z(0,0,-8,-5,3.4)),kind,d)
    assert overlap(driver,top)<1e-6 and overlap(washer,top)<1e-6
    if d==60:assert overlap(camera,robot)<1e-6 and overlap(driver,robot)<1e-6
   else:
    pcb=pose(box(7,8.6,-16,16,-16,16),kind,d)
    assert overlap(pcb,top)<1e-6
    # Check rear nut envelopes at both extremes of the M2 square-hole pattern.
    for pitch in (26,30):
     for sy in (-1,1):
      for sz in (-1,1):
       nut=cq.Workplane('YZ',origin=(-1.6,sy*pitch/2,sz*pitch/2)).polygon(6,4/math.cos(math.pi/6)).extrude(1.6)
       assert overlap(pose(nut,kind,d),top)<1e-6
   mechanical+=1
   fixture=np.concatenate([mesh_shape(top).triangles,mesh_shape(lower).triangles])
   optical=[(0,14.6)] if kind=='board' else [(h,x) for h in (10,24,40) for x in (0,20,22.5)]
   for height,forward in optical:
    local=np.array([14.6,0,0]) if kind=='board' else np.array([forward,0,height])
    origin=camera_plane(kind,d)+R@local
    count=occluded=0;H=V=0;near=1e9;far=0
    for q in (5,10,20,35,50):
     targets=np.array([[x,y,0] for x in np.linspace(60,76,5) for y in np.linspace(-q+2,q-2,9)])
     tri=np.concatenate([m.triangles for m in robot_meshes(q)]+[fixture])
     hit=blocked(origin,targets,tri);count+=len(hit);occluded+=int(hit.sum())
     h,v,n,f=field_stats(origin,R,targets);H=max(H,h);V=max(V,v);near=min(near,n);far=max(far,f)
    assert occluded==0,(kind,d,height,forward,occluded)
    axis=R[:,0];target=origin+axis*(-origin[2]/axis[2])
    cases.append(dict(kind=kind,diameter=d,lens_height=height,lens_forward=forward,lens_xyz=origin.round(2).tolist(),angle_deg=ANGLE,
      rays=count,blocked=occluded,required_hfov_deg=round(H,2),required_vfov_deg=round(V,2),focus_range_mm=[round(near,1),round(far,1)],axis_intersects_z0_at_x_mm=round(target[0],1)))
   print(kind,d,'mechanics + sightlines passed',flush=True)
 report={'mechanical_assemblies_checked':mechanical,'rays':sum(x['rays'] for x in cases),'blocked':sum(x['blocked'] for x in cases),
 'checks':'STL manifoldness; clamp screws/nuts; upper/lower clearance; 60 mm robot swept-jaw envelope; board PCB/nut envelopes; webcam 45x74x45 mm envelope; tripod washer and 50 mm driver corridor.',
 'limits':'Physical fit/load and exact camera not tested. Fixed 75 degree pitch; arbitrary lens height changes framing. Optical/mechanical report only; see arm_validation.json for full-arm conflicts and wrist clearance. No cables in model.',
 'target_region':'X=60..76, Z=0, Y inside jaw gap by 2 mm; 10/20/40/70/100 mm openings.', 'cases':cases}
 (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print('rays',report['rays'],'blocked',report['blocked'])

if __name__=='__main__':main()
