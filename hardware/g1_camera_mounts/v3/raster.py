import numpy as np
import matplotlib
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
