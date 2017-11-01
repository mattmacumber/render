import math
import sys

from random import random as rand
from PIL import Image

#render_profile = "ONE_MINUTE"
render_profile = "Infinity"
#render_profile = "SINGLE_PASS"

if render_profile == "ONE_MINUTE":
    WIDTH = 1920/4
    HEIGHT = 1024/4

    RAYS_PER_PIXEL = 4
    MAX_BOUNCE = 8
elif render_profile == "SINGLE_PASS":
    # ~ 200 seconds
    WIDTH = 1920
    HEIGHT = 1024

    RAYS_PER_PIXEL = 1
    MAX_BOUNCE = 8
else:
    WIDTH = 1920
    HEIGHT = 1024

    RAYS_PER_PIXEL = 256
    MAX_BOUNCE = 8

class V3(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = x*1.0
        self.y = y*1.0
        self.z = z*1.0
        
    def __str__(self):
        return '<v3>({},{},{})'.format(self.x,self.y,self.z)

    def __mul__(self, a):
        ret = V3()

        ret.x = a*self.x
        ret.y = a*self.y
        ret.z = a*self.z

        return ret

    def __rmul__(self, a):
        return self*a

    def __add__(self, a):
        ret = V3()

        ret.x = self.x+a.x
        ret.y = self.y+a.y
        ret.z = self.z+a.z

        return ret

    def __sub__(self, a):
        ret = V3()

        ret.x = self.x-a.x
        ret.y = self.y-a.y
        ret.z = self.z-a.z

        return ret

    def tuple(self):
        return (self.x, self.y, self.z)

def lerp(a, t, b):
    return (1.0 - t)*a + t*b;

def Hadamard(a, b):
    return V3(a.x*b.x, a.y*b.y, a.z*b.z)

def Inner(a, b):
    return a.x*b.x + a.y*b.y + a.z*b.z

def LengthSq(v3):
    return Inner(v3, v3)

def NoZ(v3):
    ret = V3()

    lensq = LengthSq(v3)
    if( lensq > (0.0001)**2):
        ret = v3 * (1.0 / math.sqrt(lensq))

    return ret

def Cross(a, b):
    ret = V3()

    ret.x = a.y*b.z - a.z*b.y
    ret.y = a.z*b.x - a.x*b.z
    ret.z = a.x*b.y - a.y*b.x

    return ret

def Linear1ToRGB255(c):
    ret = V3()
    ret.x = int(255*math.sqrt(c.x))
    ret.y = int(255*math.sqrt(c.y))
    ret.z = int(255*math.sqrt(c.z))

    return ret

def LinearToRGB(l):
    s = None

    l = max(0, l)
    l = min(1, l)

    if l <= 0.0031308:
        s = l *12.92
    else:
        s = 1.055*l**(1.0/2.4)-0.055

    return s

class World(object):
    def __init__(self):
        self.materials = list()
        self.planes = list()
        self.spheres = list()

class Sphere(object):
    def __init__(self, v3, radius, material):
        self.center = v3
        self.radius = radius
        self.material = material

class Plane(object):
    def __init__(self, n, d, material):
        self.n = n
        self.d = d
        self.material = material

class Material(object):
    def __init__(self, emit_color=V3(), refl_color=V3(), scatter=0.5):
        self.emit_color = emit_color
        self.refl_color = refl_color
        self.scatter = scatter

def RayCast(world, ray_origin, ray_dir):
    ret = V3(0, 0, 0)
    attenuation = V3(1, 1, 1)

    min_hit_distance = 0.001
    tolerance = 0.0001

    for ray_count in xrange(MAX_BOUNCE):
        hit_dist = 10**100

        hit_material = None

        next_normal = None

        for plane in world.planes:
            denom = Inner(plane.n, ray_dir)
            if denom < -tolerance or tolerance < denom:
                t = (- plane.d - Inner(plane.n, ray_origin))/denom
                if 0 < t < hit_dist:
                    hit_dist = t
                    hit_material = plane.material

                    next_normal = plane.n

        for sphere in world.spheres:
            sphere_origin_translate = ray_origin - sphere.center

            a = Inner(ray_dir, ray_dir)
            b = 2.0*Inner(ray_dir, sphere_origin_translate)
            c = Inner(sphere_origin_translate, sphere_origin_translate) - sphere.radius**2

            denom = 2*a
            sqrd = b*b-4*a*c
            sqrd = max(0, sqrd)
            root_term = math.sqrt(sqrd)
            if root_term > tolerance:

                pos = (-b + root_term) / denom
                neg = (-b - root_term) / denom

                t = pos
                if min_hit_distance < neg < pos:
                    t = neg

                if min_hit_distance < t < hit_dist:
                    hit_dist = t
                    hit_material = sphere.material

                    next_normal = NoZ(t*ray_dir+sphere_origin_translate)

        if hit_material is not None:
            ret += Hadamard(attenuation, hit_material.emit_color)
            
            cos_atten = Inner(ray_dir*-1, next_normal)
            cos_atten = max(0, cos_atten)

            attenuation = Hadamard(attenuation, cos_atten * hit_material.refl_color)

            ray_origin += hit_dist * ray_dir

            pure_bounce = ray_dir - 2*Inner(ray_dir, next_normal)*next_normal
            random_bounce = NoZ(next_normal + V3(rand()*2-1, rand()*2-1, rand()*2-1))

            ray_dir = NoZ(lerp(random_bounce, hit_material.scatter, pure_bounce))

        else:
            ret += Hadamard(attenuation, world.materials[0].emit_color)
            break 

    return ret

def render(img):
    image_width = img.size[0]
    image_height = img.size[1]

    world = World()

    world.materials.append( Material( V3(0.3, 0.4, 0.5), V3(0, 0, 0), 0.0 ))
    world.materials.append( Material( V3(0, 0, 0), V3(0.5, 0.5, 0.5), 0.0 ))
    world.materials.append( Material( V3(0, 0, 0), V3(0.7, 0.5, 0.3), 0.0 ))
    world.materials.append( Material( V3(2.0, 0.0, 0.0), V3(0, 0, 0), 0.0 ))
    world.materials.append( Material( V3(0, 0, 0), V3(0.2, 0.8, 0.2), 0.7 ))
    world.materials.append( Material( V3(0, 0, 0), V3(0.4, 0.8, 0.9), 0.85 ))
    world.materials.append( Material( V3(0, 0, 0), V3(0.95, 0.95, 0.95), 1.0 ))

    p = Plane(V3(0, 0, 1), 0, world.materials[1])
    world.planes.append(p)

    world.spheres.append(Sphere(V3(0, 0, 0), 1.0, world.materials[2]))
    world.spheres.append(Sphere(V3(3, -2, 0), 1.0, world.materials[3]))
    world.spheres.append(Sphere(V3(-2, -1, 2), 1.0, world.materials[4]))
    world.spheres.append(Sphere(V3(1, -1, 3), 1.0, world.materials[5]))
    world.spheres.append(Sphere(V3(-2, 3, 0), 2.0, world.materials[6]))

    camera_pos = V3(0, -10, 1)
    camera_z = NoZ(camera_pos)
    camera_x = NoZ(Cross(camera_z, V3(0, 0, 1)))
    camera_y = NoZ(Cross(camera_z, camera_x))

    film_dist = 1.0
    film_w = 1.0
    film_h = 1.0
    
    if image_width > image_height:
        film_h = film_w * image_height/image_width
    else:
        film_w = film_h * image_width/image_height 
   
    film_half_w = film_w/2.0
    film_half_h = film_h/2.0

    film_center = camera_pos - film_dist*camera_z

    pix_width = 0.5 / image_width
    pix_height = 0.5 / image_height

    pixels = img.load()

    for x in xrange(image_width): 
        film_x = -1.0+2.0*x/image_width;
        for y in range(image_height):
            film_y = -1.0+2.0*y/image_height;

            color = V3()
            fraction = 1.0/RAYS_PER_PIXEL
            for _ in xrange(RAYS_PER_PIXEL):

                off_x = film_x + (rand()*2-1)*pix_width
                off_y = film_y + (rand()*2-1)*pix_height

                film_p = film_center + off_x*film_half_w*camera_x + off_y*film_half_h*camera_y

                ray_origin = camera_pos
                ray_dir = NoZ(film_p - camera_pos)

                color += RayCast(world, ray_origin, ray_dir)*fraction

            pixel = V3(LinearToRGB(color.x), LinearToRGB(color.y), LinearToRGB(color.z))
            pixel = Linear1ToRGB255(pixel)
            pixels[image_width-x-1,y] = pixel.tuple()

        #if x % int(image_width/100) == 0:
        print "\rRaycasting %s %%...   " % (100.0*x/image_width),
        sys.stdout.flush()

def main():
    
    img = Image.new( 'RGB', (WIDTH, HEIGHT), "black")

    print "Rendering...",
    render(img)
    print "done."

    img.save('output.png')

    img.show()

if __name__ == "__main__":
    main()
