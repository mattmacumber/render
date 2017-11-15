#!/usr/bin/env python

import argparse
import collections
import math
import multiprocessing

from PIL import Image
from random import random
import time

""" Things you need to know to render a scene """
Render_Profile = collections.namedtuple('Render_Profile',
                                        'name width height \
                                        rays_per_pixel max_bounce')


class V3(object):
    """ Element of a 3 dimensional vector space """
    def __init__(self, x=0, y=0, z=0):
        try:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        except Exception as e:
            print e
            print x, y, z

    def __str__(self):
        return '<V3>({0.x:.3}, {0.y:.3}, {0.z:.3})'.format(self)

    def __mul__(self, a):
        ret = V3()

        ret.x = self.x*a
        ret.y = self.y*a
        ret.z = self.z*a

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
    """ linear interpolation
    0 <= t <= 1
    return a value between a and b porportional to t
    """
    return (1.0 - t)*a + t*b


def Hadamard(a, b):
    """ Hadamard product
    return the entrywise product of two inputs
    """
    return V3(a.x*b.x, a.y*b.y, a.z*b.z)


def Inner(a, b):
    """ The inner/dot product """
    return a.x*b.x + a.y*b.y + a.z*b.z


def LengthSq(v3):
    """ The square of the length of the vector """
    return Inner(v3, v3)


def NoZ(v3, e=0.0001):
    """ Normalize or Zero
    Normalize the vector if it is big enough, otherwize return the 0 vector
    """
    ret = V3()

    lensq = LengthSq(v3)
    if(lensq > e**2):
        ret = v3 * (1.0 / math.sqrt(lensq))

    return ret


def Cross(a, b):
    """ The cross product (or vector product) "a x b" """
    ret = V3()

    ret.x = a.y*b.z - a.z*b.y
    ret.y = a.z*b.x - a.x*b.z
    ret.z = a.x*b.y - a.y*b.x

    return ret


def Linear1ToRGB255(c):
    """ Map a V3(0..1) v3 to int V3(0..255) """
    ret = V3()
    ret.x = int(255*math.sqrt(c.x))
    ret.y = int(255*math.sqrt(c.y))
    ret.z = int(255*math.sqrt(c.z))

    return ret


def Gamma(linearV3):
    """ "gamma" correction for a linear V3 """

    gamma = V3(*[e*12.92 if e < 0.0031308 else 1.055*e**(1.0/2.4)-0.055
               for e in linearV3.tuple()])

    return gamma


class World(object):
    """ All of the objects and materials in a scene """
    def __init__(self):
        self.default_material = Material()
        self.planes = list()
        self.spheres = list()


class Sphere(object):
    """ Round, in 3 dimensions """
    def __init__(self, v3, radius, material):
        self.center = v3
        self.radius = radius
        self.material = material


class Plane(object):
    """ Flat, in 2 dimensions """
    def __init__(self, n, d, material):
        self.n = n
        self.d = d
        self.material = material


class Material(object):
    """ The thing things are made of """
    def __init__(self, emit_color=V3(), refl_color=V3(), scatter=0.0):
        self.emit_color = emit_color
        self.refl_color = refl_color
        self.scatter = scatter


def cast_ray(world, render_profile, ray_origin, ray_dir):
    """ Cast a ray into the world """
    result = V3(0, 0, 0)
    attenuation = V3(1, 1, 1)

    min_hit_distance = 0.001
    tolerance = 0.0001

    for _ in xrange(render_profile.max_bounce):
        hit_dist = 10**100

        hit_material = None

        next_normal = None

        for plane in world.planes:
            denom = Inner(plane.n, ray_dir)
            if abs(denom) > tolerance:
                t = (- plane.d - Inner(plane.n, ray_origin)) / denom
                if 0 < t < hit_dist:
                    hit_dist = t
                    hit_material = plane.material

                    next_normal = plane.n

        for sphere in world.spheres:
            sphere_origin_translate = ray_origin - sphere.center

            a = Inner(ray_dir, ray_dir)
            b = 2.0*Inner(ray_dir, sphere_origin_translate)
            c = Inner(sphere_origin_translate, sphere_origin_translate) \
                - sphere.radius**2

            denom = 2*a
            sqrd = max(0, b*b-4*a*c)
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

                    next_normal = NoZ(t*ray_dir + sphere_origin_translate)

        if hit_material is not None:
            result += Hadamard(attenuation, hit_material.emit_color)

            cos_atten = Inner(ray_dir*-1, next_normal)
            cos_atten = max(0, cos_atten)

            attenuation = Hadamard(attenuation, cos_atten *
                                   hit_material.refl_color)

            ray_origin += hit_dist * ray_dir

            pure_bounce = ray_dir - 2*Inner(ray_dir, next_normal)*next_normal
            random_bounce = NoZ(next_normal + V3(random()*2-1,
                                                 random()*2-1,
                                                 random()*2-1))

            ray_dir = NoZ(lerp(random_bounce,
                               hit_material.scatter,
                               pure_bounce))

        else:
            result += Hadamard(attenuation, world.default_material.emit_color)
            break

    return result


Work_Order = collections.namedtuple('Work_Order', 'world render_profile \
                                    x_min_px x_max_px y_min_px y_max_px')


def render_worker(idnum, in_queue, out_queue):
    """ Process the given work queue
    Grab an item from the work queue and render the portion of the image
    """

    while not in_queue.empty():
        try:
            work_order = in_queue.get_nowait()
        except Exception as e:
            print idnum, "Bad get in in_queue", e
            time.sleep(random.rand())
            continue

        render_profile = work_order.render_profile

        img = Image.new('RGB', (work_order.x_max_px-work_order.x_min_px,
                                work_order.y_max_px-work_order.y_min_px),
                        "blue")

        camera_pos = V3(0, -10, 1)
        camera_z = NoZ(camera_pos)
        camera_x = NoZ(Cross(camera_z, V3(0, 0, 1)))
        camera_y = NoZ(Cross(camera_z, camera_x))

        image_width = render_profile.width
        image_height = render_profile.height

        film_dist = 1.0
        film_w = 1.0
        film_h = 1.0

        # Match the film aspect ratio to match the image
        if image_width > image_height:
            film_h = film_w * image_height/image_width
        else:
            film_w = film_h * image_width/image_height

        film_center = camera_pos - film_dist*camera_z

        pix_width = 1.0 / image_width
        pix_height = 1.0 / image_height

        pixels = img.load()

        for x in xrange(work_order.x_min_px, work_order.x_max_px):
            film_x = -1.0+2.0*x/image_width
            for y in range(work_order.y_min_px, work_order.y_max_px):
                film_y = -1.0+2.0*y/image_height

                color = V3()

                # Cast multiple rays and composite them equally
                fraction = 1.0/render_profile.rays_per_pixel
                for _ in xrange(render_profile.rays_per_pixel):

                    # add a < 1 px jitter to each ray
                    off_x = film_x + (random()*2-1)*pix_width/2.0
                    off_y = film_y + (random()*2-1)*pix_height/2.0

                    film_p = film_center - off_x*film_w/2.0*camera_x + \
                        off_y * film_h/2.0 * camera_y

                    ray_origin = camera_pos
                    ray_dir = NoZ(film_p - camera_pos)

                    result = cast_ray(work_order.world, render_profile,
                                      ray_origin, ray_dir)

                    color += result*fraction
                pixel = Gamma(color)
                pixel = Linear1ToRGB255(pixel)
                try:
                    pixels[x-work_order.x_min_px,
                           y-work_order.y_min_px] = pixel.tuple()
                except Exception as e:
                    print e

        out_queue.put((work_order, img))


def load_world():
    """ Return a populated world object """

    world = World()

    p = Plane(V3(0, 0, 1), 0, Material(V3(0, 0, 0), V3(0.5, 0.5, 0.5), 0.0))
    world.planes.append(p)

    world.default_material = Material(V3(0.3, 0.4, 0.5), V3(0, 0, 0), 0.0)

    world.spheres.append(Sphere(V3(0, 0, 0), 1.0,
                                Material(V3(0, 0, 0), V3(0.7, 0.5, 0.3), 0.0)))
    world.spheres.append(Sphere(V3(3, -2, 0), 1.0,
                                Material(V3(2.0, 0.0, 0.0), V3(0, 0, 0), 0.0)))
    world.spheres.append(Sphere(V3(-2, -1, 2), 1.0,
                                Material(V3(0, 0, 0), V3(0.2, 0.8, 0.2), 0.7)))
    world.spheres.append(Sphere(V3(1, -1, 3), 1.0,
                                Material(V3(0, 0, 0), V3(0.4, 0.8, 0.9), 0.85)))
    world.spheres.append(Sphere(V3(-2, 3, 0), 2.0,
                                Material(V3(0, 0, 0),
                                         V3(0.95, 0.95, 0.95), 1.0)))

    return world


def render(profile, thread_count):
    """ Use the given render profile and thread cound to render """

    img = Image.new('RGB', (profile.width, profile.height), "black")

    world = load_world()

    start_time = time.time()

    # Set the tile width to be a power of two
    tile_width = 1
    while 2*tile_width <= img.size[0]/math.sqrt(thread_count):
        tile_width *= 2

    tile_height = tile_width
    tile_count_x = (img.size[0] + tile_width - 1) / tile_width
    tile_count_y = (img.size[1] + tile_width - 1) / tile_height

    print "Chunking: %d threads with %d %dx%d tiles" % \
        (thread_count, tile_count_x*tile_count_y, tile_width, tile_height)

    job_queue = multiprocessing.Queue()
    image_queue = multiprocessing.Queue()

    for tile_x in xrange(tile_count_x):
        x_min = tile_x*tile_width
        x_max = min(img.size[0], x_min + tile_width)
        for tile_y in xrange(tile_count_y):
            y_min = tile_y*tile_height
            y_max = min(img.size[1], y_min + tile_height)

            work_order = Work_Order(world, profile, x_min, x_max, y_min, y_max)
            job_queue.put(work_order)

    procs = list()
    for n in xrange(thread_count):
        proc = multiprocessing.Process(target=render_worker,
                                       args=(n, job_queue, image_queue))
        proc.start()
        procs.append(proc)

    for x in xrange(tile_count_x):
        for y in xrange(tile_count_y):
            work_order, tile = image_queue.get()
            img.paste(tile, (work_order.x_min_px, work_order.y_min_px))

    end_time = time.time()
    casting_time = (end_time - start_time)*1000

    print "Raycasting time: %dms" % casting_time

    return img


def main():

    parser = argparse.ArgumentParser(description='A Simple Ray Tracer')
    parser.add_argument('-t', '--threads', type=int,
                        default=multiprocessing.cpu_count(), nargs='?',
                        help='Number of threads to use')
    parser.add_argument('-y', '--height', type=int, default=1080/4, nargs='?',
                        help='Image Height in pixels')
    parser.add_argument('-x', '--width', type=int, default=1920/4, nargs='?',
                        help='Image Width in pixels')
    parser.add_argument('-rpp', '--rays_per_pixel', type=int, default=1,
                        nargs='?', help='Image Height in pixels')
    parser.add_argument('-mb', '--max_bounce', type=int, default=4, nargs='?',
                        help='Image Width in pixels')

    args = parser.parse_args()

    profile = Render_Profile('test', args.width, args.height,
                             args.rays_per_pixel, args.max_bounce)

    print "Rendering...",
    img = render(profile, args.threads)
    print "done."

    img.save('output.png')

    img.show()


if __name__ == "__main__":
    main()
