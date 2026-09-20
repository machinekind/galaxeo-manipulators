"""Check nominal fastener envelopes and screwdriver approach space."""
from design import *
from verify import overlap,robot_envelope

def hex_z(x,y,z,af,h):
 return cq.Workplane('XY',origin=(x,y,z)).polygon(6,af/math.cos(math.pi/6)).extrude(h)

def hex_y(x,z,y0,af,h):
 return cq.Workplane('XZ',origin=(x,y0+h,z)).polygon(6,af/math.cos(math.pi/6)).extrude(h)


def check():
 checks=0
 for d in DIMS:
  upper=clamp(d);lower=clamp(d,False);deck=deck_z(d)
  for y in (-10,10):
   for x in (17,41):
    nut=hex_z(x,y,deck-8-3.2,7,3.2)
    assert overlap(nut,upper)<1e-6,('boom nut',d,x,y)
    checks+=1
  for sy in (-1,1):
   y=sy*((d+.6)/2+10)
   nut=hex_z(0,y,-8.9,7,3.2)
   shaft=cyl_z(0,y,-10.2,9.8,2)
   assert overlap(nut,lower)<1e-6
   assert overlap(shaft,upper)<1e-6 and overlap(shaft,lower)<1e-6
   checks+=2
 # M4x20 boom screws including washer/head clear the robot envelope.
 robot=robot_envelope()
 for x in (17,41):
  for y in (-10,10):
   tip=deck_z(60)+6+.8-20
   screw=cyl_z(x-45,y,tip,deck_z(60)+6+.8,2)
   assert overlap(screw,robot)<1e-6
   checks+=1
 for kind,carrier in [('board',board_carrier()),('webcam',webcam_carrier())]:
  for s in (-1,1):
   for x,af,h in [(0,7,3.2),(-25,5.5,2.4)]:
    y0=40 if s==1 else -40-h
    nut=hex_y(x,0,y0,af,h)
    assert overlap(nut,carrier)<1e-6,(kind,s,x,'nut')
    checks+=1
  for angle in (35,55,75):
   for slide in SLIDE:
    # 50 mm long x 10 mm diameter driver corridor below tripod screw.
    if kind=='webcam':
     driver=cyl_z(0,0,-81,-30.1,5)
     driver=pose_carrier(driver,60,slide,angle)
     assert overlap(driver,pose_boom(boom(),60,slide))<1e-6
     assert overlap(driver,clamp(60).translate((-45,0,0)))<1e-6
     assert overlap(driver,robot)<1e-6
     checks+=1
 report={'fastener_envelope_checks':checks,'results':'Nominal hex nuts fit pockets; boom nuts clear ribs; M4 boom screws clear the modeled robot; a 50 mm driver corridor below the tripod slot is clear in six tested poses.',
 'assumptions':'M4 nut AF 7 mm, height 3.2 mm; M3 AF 5.5 mm, height 2.4 mm; specified washers and printed 2 mm hinge spacers. Real hardware tolerances and webcam socket depth must be checked.'}
 (ROOT/'fastener_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(report,flush=True)
if __name__=='__main__':check()
