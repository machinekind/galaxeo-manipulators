import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/g1-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from design import *
from geometry_checks import robot_meshes
from raster import raster
BG='#f4f6f8';INK='#153a49'

def scene(kind,with_camera=True):
 items=[(m,'#83919b') for m in robot_meshes(20)]
 items += [(mesh_shape(upper(kind,60)),'#de9b39'),(mesh_shape(clamp(60,False)),'#15888a')]
 if with_camera:
  if kind=='board':
   items += [(mesh_shape(pose(box(7,8.6,-16,16,-16,16),kind,60)),'#327a61'),(mesh_shape(pose(cq.Workplane('YZ',origin=(8.6,0,0)).circle(6).extrude(8),kind,60)),'#163245')]
  else:
   items += [(mesh_shape(pose(box(-20,20,-32,32,0,36).edges('|Z').fillet(3),kind,60)),'#253c4e'),(mesh_shape(pose(cq.Workplane('YZ',origin=(20,0,24)).circle(7).extrude(2),kind,60)),'#397083')]
 return items

def render_iso(ax,items):
 triangles=[];colors=[]
 for m,col in items:
  shade=.57+.43*np.abs(m.face_normals@np.array([.4,-.5,.768]));colors.extend(shade[:,None]*np.array(matplotlib.colors.to_rgb(col)));triangles.extend(m.triangles)
 ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,edgecolors='none',zsort='average'))
 ax.set_proj_type('ortho');ax.set(xlim=(-85,85),ylim=(-85,85),zlim=(-35,135));ax.set_box_aspect((1,1,1));ax.view_init(16,-46);ax.axis('off')

def main():
 fig=plt.figure(figsize=(14,10),facecolor=BG)
 fig.text(.045,.945,'FIXED 60° / TWO MAIN PRINTED PARTS',fontsize=23,weight='bold',color=INK)
 fig.text(.045,.91,'Integrated camera bracket + lower clamp. Two M4 clamp screws. No sliding or tilt mechanisms.',fontsize=12,color='#526c7b')
 for i,kind in enumerate(('board','webcam')):
  fig.text(.07+i*.5,.85,'32 mm BOARD CAMERA' if kind=='board' else 'TRIPOD WEBCAM',fontsize=14,weight='bold',color=INK)
  ax=fig.add_axes([.025+.5*i,.46,.46,.38],projection='3d',facecolor=BG);render_iso(ax,scene(kind))
  R=rotation();local=np.array([14.6,0,0]) if kind=='board' else np.array([20,0,24]);eye=camera_plane(kind,60)+R@local
  target=eye+R[:,0]*(-eye[2]/R[2,0])
  ax.plot([eye[0],target[0]],[0,0],[eye[2],0],color='#ae532d',linestyle='--',lw=1.8)
  ax.scatter([target[0]],[0],[0],s=18,color='#ae532d')
  items=scene(kind,False)
  for x in range(-100,221,20):
   for y in range(-160,161,20):items.append((mesh_shape(box(x,x+19.5,y,y+19.5,-31,-30)),'#d7e1df' if (x//20+y//20)%2 else '#e5eaea'))
  im,depth,fx,fy=raster(items,eye,R)
  for x in np.linspace(60,76,5):
   for y in np.linspace(-18,18,9):
    p=(np.array([x,y,0])-eye)@R;u=int(400-p[1]*fx/p[0]);v=int(300-p[2]*fy/p[0])
    if 3<=u<797 and 3<=v<597 and p[0]<=depth[v,u]+.1:im[v-2:v+3,u-2:u+3]=[10,154,126]
  Image.fromarray(im).save(ROOT/f'{kind}_camera_view.png')
  ax2=fig.add_axes([.065+.5*i,.125,.37,.30]);ax2.imshow(im);ax2.axis('off')
 fig.text(.07,.45,'Simulated view from the fixed board camera',fontsize=11,color=INK)
 fig.text(.57,.45,'Simulated view with a nominal webcam',fontsize=11,color=INK)
 fig.text(.045,.065,'Green samples: checked grasp region. Simulated lens: 70° × 55°. Physical fit, camera focus and load remain untested.',fontsize=10,color='#526c7b')
 fig.savefig(ROOT/'preview_fixed.png',dpi=160,facecolor=BG)
 print(ROOT/'preview_fixed.png')
if __name__=='__main__':main()
