"""Depth-buffered renders including actual A1 arm meshes, not a gripper-only sketch."""
import os,importlib.util
os.environ.setdefault('MPLCONFIGDIR','/tmp/g1-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from arm_check import *
from raster import raster
BG='#e6edf0';INK='#173b4c'
FOLDED=np.radians([-111.7682,105.9152,-1.9636,9.5397,-66.4478,108.1904])
NORMAL=np.radians([0,90,-120,30,0,0])

def camera_axes(eye,target):
 f=np.array(target)-eye;f/=np.linalg.norm(f)
 left=np.cross([0.,0.,1.],f);left/=np.linalg.norm(left)
 return np.column_stack([f,left,np.cross(f,left)])

def robot_items(robot,q):
 frames=robot.frames(q,20);items=[]
 for name,m in robot.meshes.items():
  copy=m.copy();copy.apply_transform(frames[name])
  col='#778998' if name=='arm_link2' else '#aab6c0'
  if 'gripper' in name:col='#718996'
  items.append((copy,col))
 return items

def payload_items(module,kind):
 items=[(module.mesh_shape(module.upper(kind,60)),'#dc942f'),(module.mesh_shape(module.clamp(60,False)),'#158d8c')]
 if kind=='board':
  items.extend([(module.mesh_shape(module.pose(module.box(7,8.6,-16,16,-16,16),kind,60)),'#267650'),(module.mesh_shape(module.pose(cq.Workplane('YZ',origin=(8.6,0,0)).circle(6).extrude(8),kind,60)),'#1c3747')])
 else:
  items.extend([(module.mesh_shape(module.pose(module.box(-22.5,22.5,-37,37,0,45),kind,60)),'#263e51'),(module.mesh_shape(module.pose(cq.Workplane('YZ',origin=(22.5,0,24)).circle(7).extrude(.3),kind,60)),'#318596')])
 return items

def take(items,eye,target,path,w=1000,h=700,hfov=55,vfov=40):
 eye=np.array(eye,dtype=float);R=camera_axes(eye,target)
 im,*_=raster(items,eye,R,w=w,h=h,hfov=hfov,vfov=vfov)
 Image.fromarray(im).save(path)
 return im

def old_design():
 spec=importlib.util.spec_from_file_location('revision3',ROOT.parent/'v3/design.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod

def mark_collision(im,point):
 eye=np.array([160.,-320.,185.]);R=camera_axes(eye,[-35,0,35]);p=(point-eye)@R
 h,w=im.shape[:2];fx=w/2/np.tan(np.radians(55/2));fy=h/2/np.tan(np.radians(40/2))
 u=w/2-p[1]*fx/p[0];v=h/2-p[2]*fy/p[0]
 pic=Image.fromarray(im);draw=ImageDraw.Draw(pic)
 draw.ellipse((u-30,v-30,u+30,v+30),outline='#d72e3a',width=4)
 draw.line((u-28,v-15,u-85,v-52),fill='#d72e3a',width=4)
 im=np.array(pic);pic.save(ROOT/'old_collision.png');return im

def main():
 import design as current
 robot=Robot();old=old_design();robot.set_pose(FOLDED)
 assert not robot.bare_collisions()
 comparison={'q_degrees':np.rad2deg(FOLDED).tolist(),'jaw_half_gap_mm':20,'old':{},'new':{}}
 for kind in ('board','webcam'):
  oldparts={'upper':collision_object(old.mesh_shape(old.upper(kind,60))),'lower':collision_object(old.mesh_shape(old.clamp(60,False)))}
  oldhits=mount_collisions(robot,oldparts);newhits=mount_collisions(robot,accessory_objects(kind))
  assert oldhits and not newhits,(kind,oldhits,newhits)
  minimum=1e9
  for obj in accessory_objects(kind).values():
   for name in robot.names[:-1]:minimum=min(minimum,fcl.distance(obj,robot.objects[name],fcl.DistanceRequest(),fcl.DistanceResult()))
  comparison['old'][kind]=oldhits;comparison['new'][kind]={'collisions':newhits,'minimum_arm_clearance_mm':round(minimum,3)}
 (ROOT/'comparison_pose.json').write_text(json.dumps(comparison,indent=2)+'\n')
 # Same pose, camera and scale on both sides of the comparison.
 contacts=[]
 before=robot_items(robot,FOLDED)+payload_items(old,'webcam')
 for part in (old.upper('webcam',60),old.clamp(60,False)):
  result=fcl.CollisionResult()
  fcl.collide(collision_object(old.mesh_shape(part)),robot.objects['arm_link2'],fcl.CollisionRequest(num_max_contacts=8,enable_contact=True),result)
  for contact in result.contacts:
   contacts.append(contact.pos)
   mark=cq.Workplane('XY').sphere(2.2).translate(tuple(contact.pos));before.append((mesh_shape(mark),'#dc3a3f'))
 after=robot_items(robot,FOLDED)+payload_items(current,'webcam')
 oldim=take(before,[160,-320,185],[-35,0,35],ROOT/'old_collision.png')
 oldim=mark_collision(oldim,np.mean(contacts,axis=0))
 newim=take(after,[160,-320,185],[-35,0,35],ROOT/'new_same_pose.png')
 details=[]
 for kind in ('board','webcam'):
  items=robot_items(robot,NORMAL)+payload_items(current,kind)
  details.append(take(items,[165,-340,190],[-65,0,35],ROOT/f'{kind}_on_arm.png'))
  # Export the complete actual-arm assembly for a 3D viewer, in metres.
  glb=trimesh.Scene()
  for i,(m,col) in enumerate(items):
   mm=m.copy();mm.apply_scale(.001);mm.visual.face_colors=np.array(matplotlib.colors.to_rgba(col))*255
   glb.add_geometry(mm,node_name=f'part_{i}')
  glb.export(ROOT/f'{kind}_on_arm.glb')
  R=rotation();local=np.array([14.6,0,0]) if kind=='board' else np.array([22.5,0,24])
  eye=camera_plane(kind,60)+R@local
  # Exclude the camera itself, include the full arm and both clamp halves.
  optical=robot_items(robot,NORMAL)+[(mesh_shape(upper(kind,60)),'#dc942f'),(mesh_shape(clamp(60,False)),'#158d8c')]
  im,*_=raster(optical,eye,R,hfov=70,vfov=55)
  Image.fromarray(im).save(ROOT/f'{kind}_camera_view.png')
 compose(oldim,newim,details)

def compose(oldim,newim,details):
 fig=plt.figure(figsize=(14,10),facecolor=BG)
 fig.text(.04,.955,'ARM CLEARANCE: BEFORE / REVISED',fontsize=23,weight='bold',color=INK)
 fig.text(.04,.919,'Same folded arm pose. Red circle locates the previous clamp/arm collision, partly hidden by the arm housing.',fontsize=11,color=INK)
 panels=[(oldim,'V3: upper and lower clamp hit the upper arm'),(newim,'V4: 13.6 mm minimum arm clearance in this pose'),(details[0],'Board camera — fixed 75° bracket on the real wrist'),(details[1],'Webcam — same fixed angle, accessible tripod screw')]
 for i,(im,title) in enumerate(panels):
  x=.035+.5*(i%2);y=.52 if i<2 else .13
  ax=fig.add_axes([x,y,.46,.33]);ax.imshow(im);ax.axis('off')
  fig.text(x,y+.345,title,fontsize=11,weight='bold',color=INK)
 fig.text(.04,.055,'Local wrist clearance is verified. Other folded-arm poses still collide: include the payload in motion collision checks.',fontsize=11,color='#983a2d')
 fig.savefig(ROOT/'arm_clearance.png',dpi=160,facecolor=BG)
 print(ROOT/'arm_clearance.png')

if __name__=='__main__':main()
