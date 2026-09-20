"""Render the actual CAD geometry, plus labelled illustrative camera envelopes."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/g1-mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import cadquery as cq
from generate import ROOT, clamp, board_adapter, webcam_adapter, deck_height, box, CAMERA_TILT


def mesh(part):
    v,f=part.val().tessellate(.12,.15)
    return trimesh.Trimesh(vertices=[p.toTuple() for p in v],faces=f,process=False)


def draw(ax,m,color,alpha=1):
    rgb=np.array(matplotlib.colors.to_rgb(color))
    shade=.62+.38*np.maximum(0,m.face_normals@np.array([-.3,-.4,.866]))
    ax.add_collection3d(Poly3DCollection(m.triangles,facecolors=shade[:,None]*rgb,
                                       edgecolors="none",alpha=alpha,zsort="average"))


def cad(ax,p,color):
    draw(ax,mesh(p),color)


def configure(ax,center,span,elev=24,azim=-48):
    ax.set_proj_type('ortho')
    ax.set(xlim=(center[0]-span/2,center[0]+span/2),
           ylim=(center[1]-span/2,center[1]+span/2),
           zlim=(center[2]-span/2,center[2]+span/2))
    ax.set_box_aspect((1,1,1));ax.view_init(elev,azim);ax.axis('off')


fig=plt.figure(figsize=(15,10),facecolor='#f3f5f7')
fig.text(.045,.95,'G1 / WRIST CAMERA MOUNTS',fontsize=24,weight='bold',color='#19394b')
fig.text(.045,.918,'Printable prototypes  •  shared split clamp  •  two interchangeable adapters',fontsize=12,color='#526773')
repo=ROOT.parents[1]
meshes=repo/'ros2_ws/src/galaxea_a1xy_description/meshes'
for col,kind in enumerate(('board','webcam')):
    ax=fig.add_axes([.02+col*.5,.39,.48,.50],projection='3d',facecolor='#f3f5f7')
    body=trimesh.load(meshes/'gripper_link.STL');body.apply_scale(1000);draw(ax,body,'#6d7a86')
    for i,sgn in ((1,1),(2,-1)):
        finger=trimesh.load(meshes/f'gripper_finger_link{i}.STL');finger.apply_scale(1000)
        finger.apply_translation([36.89,sgn*(13.453+20),sgn*.12059]);draw(ax,finger,'#8e9ba7')
    cad(ax,clamp(60,True).translate((-45,0,0)),'#138f96')
    cad(ax,clamp(60,False).translate((-45,0,0)),'#138f96')
    adapter=board_adapter() if kind=='board' else webcam_adapter()
    cad(ax,adapter.translate((-45,0,deck_height(60))),'#edaa42')
    if kind=='board':
        def pose(p):
            return p.translate((0,0,20)).rotate((0,0,0),(0,1,0),CAMERA_TILT).translate((24-45,0,2+deck_height(60)))
        pcb=box(7,8.6,-16,16,-16,16)
        cad(ax,pose(pcb),'#29594d')
        lens=cq.Workplane('YZ',origin=(8.6,0,0)).circle(7).extrude(10)
        cad(ax,pose(lens),'#172c3b')
    else:
        # Illustrative small webcam, not a model-specific fit guarantee.
        cam=box(27,57,-19,19,6,29).edges('|Z').fillet(5)
        cad(ax,cam.translate((-45,0,deck_height(60))),'#263c4c')
        lens=cq.Workplane('YZ',origin=(57,0,18)).circle(7).extrude(3)
        cad(ax,lens.translate((-45,0,deck_height(60))),'#4c9cad')
    configure(ax,(-5,0,18),170)
    fig.text(.06+col*.5,.855,'01   BOARD CAMERA' if kind=='board' else '02   TRIPOD WEBCAM',fontsize=14,weight='bold',color='#19394b')
    fig.text(.06+col*.5,.435,'32 × 32 mm board · 26–30 mm hole pitch · 20° tilt' if kind=='board' else '¼″-20 screw · 14 mm travel · 52 × 48 mm shelf',fontsize=11,color='#526773')
    ax2=fig.add_axes([.09+col*.5,.11,.33,.27],projection='3d',facecolor='#f3f5f7')
    cad(ax2,adapter,'#edaa42');b=adapter.val().BoundingBox()
    configure(ax2,((b.xmin+b.xmax)/2,0,(b.zmin+b.zmax)/2),65,elev=28,azim=-48)
fig.text(.045,.065,'PRINT PER MOUNT: 1 upper clamp + 1 matching lower clamp + 1 camera adapter',fontsize=12,weight='bold',color='#19394b')
fig.text(.045,.035,'57 mm and 60 mm clamp options included. Check the fit gauge first. Camera shapes are illustrative; physical fit/load untested.',fontsize=10,color='#526773')
fig.savefig(ROOT/'preview.png',dpi=170,facecolor=fig.get_facecolor())
print(ROOT/'preview.png')
