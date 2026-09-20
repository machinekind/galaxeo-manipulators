"""CAD overview, side sightlines and a synthetic pinhole camera view."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/g1-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import numpy as np
from design import *
from verify import robot_meshes,camera_pose,rotation

BG='#f4f6f8';INK='#17384a';TEAL='#15888a';GOLD='#e8a33d'

def fixtures(kind,angle):
 d=60;slide=8
 return [(mesh_shape(clamp(d).translate((-45,0,0))),TEAL),
 (mesh_shape(clamp(d,False).translate((-45,0,0))),TEAL),
 (mesh_shape(pose_boom(boom(),d,slide)),TEAL),
 (mesh_shape(pose_carrier(board_carrier() if kind=='board' else webcam_carrier(),d,slide,angle)),GOLD)]

def plot_3d(ax,items):
 triangles=[];colors=[]
 light=np.array([.4,-.5,.768])
 for m,color in items:
  rgb=np.array(matplotlib.colors.to_rgb(color));shade=.55+.45*np.abs(m.face_normals@light)
  triangles.extend(m.triangles);colors.extend(shade[:,None]*rgb)
 # All faces sorted together: correct occlusion between separately modelled parts.
 ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,edgecolors='none',zsort='average'))
 ax.set_proj_type('ortho');ax.set(xlim=(-90,90),ylim=(-90,90),zlim=(-35,145));ax.set_box_aspect((1,1,1));ax.view_init(19,-49);ax.axis('off')


def raster(items,eye,R,w=800,h=600,hfov=70,vfov=55):
 rgb=np.full((h,w,3),[230,237,240],dtype=np.uint8);depth=np.full((h,w),np.inf)
 fx=w/2/np.tan(np.radians(hfov/2));fy=h/2/np.tan(np.radians(vfov/2))
 for mesh,col in items:
  q=(mesh.vertices-eye)@R
  uv=np.column_stack((w/2-q[:,1]*fx/q[:,0],h/2-q[:,2]*fy/q[:,0]))
  for fi,face in enumerate(mesh.faces):
   z=q[face,0]
   if min(z)<1:continue
   t=uv[face];xmin=max(0,int(np.floor(t[:,0].min())));xmax=min(w-1,int(np.ceil(t[:,0].max())))
   ymin=max(0,int(np.floor(t[:,1].min())));ymax=min(h-1,int(np.ceil(t[:,1].max())))
   if xmax<xmin or ymax<ymin:continue
   x0,y0=t[0];x1,y1=t[1];x2,y2=t[2]
   den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
   if abs(den)<1e-8:continue
   yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1];xx=xx+.5;yy=yy+.5
   a=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
   b=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;c=1-a-b
   inside=(a>=-1e-7)&(b>=-1e-7)&(c>=-1e-7)
   zz=1/(a/z[0]+b/z[1]+c/z[2]);old=depth[ymin:ymax+1,xmin:xmax+1]
   mask=inside&(zz<old);old[mask]=zz[mask]
   light=.65+.35*abs(mesh.face_normals[fi]@np.array([.3,-.4,.866]))
   rgb[ymin:ymax+1,xmin:xmax+1][mask]=np.array(matplotlib.colors.to_rgb(col))*255*light
 return rgb,depth,fx,fy


def main():
 fig=plt.figure(figsize=(16,12),facecolor=BG)
 fig.text(.04,.955,'G1 CAMERA MOUNTS / REVISION 2',fontsize=25,weight='bold',color=INK)
 fig.text(.04,.924,'Aim at the grasp · 30 mm sliding travel · 35–75° tilt · accessible camera fasteners',fontsize=13,color='#526c7b')
 for col,kind in enumerate(('board','webcam')):
  eye,R,angle=camera_pose(kind)
  ax=fig.add_axes([.025+.50*col,.48,.45,.41],projection='3d',facecolor=BG)
  items=[(m,'#727f8b') for m in robot_meshes(20)]+fixtures(kind,angle)
  if kind=='board':
   pcb=box(7,8.6,-16,16,-16,16);lens=cq.Workplane('YZ',origin=(8.6,0,0)).circle(6).extrude(8)
   items += [(mesh_shape(pose_carrier(pcb,angle=angle)),'#327961'),(mesh_shape(pose_carrier(lens,angle=angle)),'#142a3a')]
  else:
   body=box(-20,20,-32,32,-24,12).edges('|Z').fillet(3)
   lens=cq.Workplane('YZ',origin=(20,0,0)).circle(7).extrude(2)
   items += [(mesh_shape(pose_carrier(body,angle=angle)),'#263d4f'),(mesh_shape(pose_carrier(lens,angle=angle)),'#37697b')]
  plot_3d(ax,items)
  ax.plot([eye[0],75],[0,0],[eye[2],0],color='#ca6036',lw=2,linestyle='--')
  ax.scatter([75],[0],[0],color='#ca6036',s=30)
  fig.text(.05+.5*col,.865,'01  BOARD CAMERA' if kind=='board' else '02  TRIPOD WEBCAM',fontsize=14,weight='bold',color=INK)
  fig.text(.05+.5*col,.485,'32 mm board · four slotted M2 mounts' if kind=='board' else '¼″-20 screw from below · tilting cradle',fontsize=12,color='#526c7b')
 # Side section/sightline figure.
 ax=fig.add_axes([.06,.12,.40,.29],facecolor='white')
 for m in robot_meshes(20):
  ax.add_collection(PolyCollection(m.triangles[:,:,[0,2]],facecolors='#a6b2bc',edgecolors='none',alpha=.7))
 eye,R,a=camera_pose('board');ax.plot([eye[0],75],[eye[2],0],color='#c76539',lw=2.5)
 ax.scatter([eye[0],75],[eye[2],0],c=['#17384a','#c76539'],s=[55,35],zorder=5)
 for target in [[60,0,0],[76,0,0]]:ax.plot([eye[0],target[0]],[eye[2],0],c='#188b88',lw=1)
 ax.annotate('Lens',eye[[0,2]],xytext=(-32,123),arrowprops={'arrowstyle':'-','color':INK},color=INK)
 ax.annotate('Grasp region',xy=(70,0),xytext=(110,23),arrowprops={'arrowstyle':'-','color':INK},color=INK)
 ax.text(32,79,f'{a:.1f}° down\n~119 mm to target',color='#a04b27',fontsize=11)
 ax.set(xlim=(-85,160),ylim=(-35,145),xlabel='Forward from gripper body [mm]',ylabel='Height above gripper axis [mm]')
 ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.text(.06,.435,'SIGHTLINE THROUGH THE FINGERTIP GAP',fontsize=13,weight='bold',color=INK)
 # Actual depth-rasterized pinhole image, not a decorative render.
 scene=[(m,'#8e9da7') for m in robot_meshes(20)]+fixtures('board',a)
 # Ground plane is 30 mm below the gripper axis; grid clarifies camera orientation.
 for x in range(-100,221,20):
  for y in range(-160,161,20):
   ground=box(x,x+19.5,y,y+19.5,-31,-30)
   scene.append((mesh_shape(ground),'#e4e9e9' if (x//20+y//20)%2 else '#d6e0df'))
 im,depth,fx,fy=raster(scene,eye,R)
 # Overlay the 45 verified target samples at q=20 mm, using depth-tested centres.
 for x in np.linspace(60,76,5):
  for y in np.linspace(-18,18,9):
   p=(np.array([x,y,0])-eye)@R;u=int(400-p[1]*fx/p[0]);v=int(300-p[2]*fy/p[0])
   if 3<=u<797 and 3<=v<597 and p[0]<=depth[v,u]+.1:im[v-2:v+3,u-2:u+3]=[10,154,126]
 Image.fromarray(im).save(ROOT/'camera_view_assumed_70x55.png')
 ax2=fig.add_axes([.55,.105,.39,.31]);ax2.imshow(im);ax2.axis('off')
 fig.text(.55,.435,'SIMULATED CAMERA VIEW',fontsize=13,weight='bold',color=INK)
 fig.text(.55,.079,'Assumed 70° × 55° lens; 40 mm jaw opening.\nGreen samples are the checked grasp region.',fontsize=10,color='#526c7b')
 fig.text(.04,.045,'Geometry and sightlines checked against repository G1 meshes. Physical fit, focus and load still need a hardware check.',fontsize=11,color='#526c7b')
 fig.savefig(ROOT/'preview_v2.png',dpi=160,facecolor=BG)
 print(ROOT/'preview_v2.png')

if __name__=='__main__':main()
