#!/usr/bin/env python

import argparse
import math
import sys
import threading

from PIL import Image
from random import random
from time import time

render_profile = "QUICK_SMALL"
# render_profile = "PRETTY_SMALL"
# render_profile = "SINGLE_PASS"
# render_profile = "Infinity"

if render_profile == "QUICK_SMALL":
    WIDTH = 1920/4
    HEIGHT = 1024/4

    RAYS_PER_PIXEL = 4
    MAX_BOUNCE = 4
elif render_profile == "SINGLE_PASS":
    WIDTH = 1920
    HEIGHT = 1024

    RAYS_PER_PIXEL = 1
    MAX_BOUNCE = 4
elif render_profile == "PRETTY_SMALL":
    WIDTH = 1920/4
    HEIGHT = 1024/4

    RAYS_PER_PIXEL = 10
    MAX_BOUNCE = 4
else:
    WIDTH = 1920
    HEIGHT = 1024

    RAYS_PER_PIXEL = 256
    MAX_BOUNCE = 8


class V3(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

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
    return (1.0 - t)*a + t*b


def Hadamard(a, b):
    return V3(a.x*b.x, a.y*b.y, a.z*b.z)


def Inner(a, b):
    return a.x*b.x + a.y*b.y + a.z*b.z


def LengthSq(v3):
    return Inner(v3, v3)


def NoZ(v3):
    ret = V3()

    lensq = LengthSq(v3)
    if(lensq > (0.0001)**2):
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


def LinearToRGB(linear):
    gamma = None

    linear = max(0, linear)
    linear = min(1, linear)

    if linear <= 0.0031308:
        gamma = linear * 12.92
    else:
        gamma = 1.055*linear**(1.0/2.4)-0.055

    return gamma


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
    def __init__(self, emit_color=V3(), refl_color=V3(), scatter=0.0):
        self.emit_color = emit_color
        self.refl_color = refl_color
        self.scatter = scatter


class WorkQueue(object):
    def __init__(self):
        self.work_orders = list()
        self.bounces_computed = 0
        self.tiles_rendered = 0
        self.total_tile_count = 0


def cast_ray(world, ray_origin, ray_dir):
    result = V3(0, 0, 0)
    attenuation = V3(1, 1, 1)

    min_hit_distance = 0.001
    tolerance = 0.0001

    bounces_computed = 0

    for bounce_count in xrange(MAX_BOUNCE):
        bounces_computed += 1
        hit_dist = 10**100

        hit_material = None

        next_normal = None

        for plane in world.planes:
            denom = Inner(plane.n, ray_dir)
            if denom < -tolerance or tolerance < denom:
                t = (- plane.d - Inner(plane.n, ray_origin)) / \
                    denom
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

                    next_normal = NoZ(t*ray_dir + sphere_origin_translate)

        if hit_material is not None:
            result += Hadamard(attenuation, hit_material.emit_color)

            cos_atten = Inner(ray_dir*-1, next_normal)
            cos_atten = max(0, cos_atten)

            attenuation = Hadamard(attenuation, cos_atten *
                                   hit_material.refl_color)

            ray_origin += hit_dist * ray_dir

            pure_bounce = ray_dir - \
                2*Inner(ray_dir, next_normal)*next_normal
            random_bounce = NoZ(next_normal +
                                V3(random()*2-1,
                                    random()*2-1,
                                    random()*2-1))

            ray_dir = NoZ(lerp(random_bounce,
                               hit_material.scatter,
                               pure_bounce))

        else:
            result += Hadamard(attenuation, world.materials[0].emit_color)
            break

    return result


def render_tile(queue):

    while len(queue.work_orders) > 0:
        work_order = queue.work_orders.pop()
        world, image, x_min, y_min, x_max, y_max = work_order

        camera_pos = V3(0, -10, 1)
        camera_z = NoZ(camera_pos)
        camera_x = NoZ(Cross(camera_z, V3(0, 0, 1)))
        camera_y = NoZ(Cross(camera_z, camera_x))

        image_width = image.size[0]
        image_height = image.size[1]

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

        pixels = image.load()

        bounces_computed = 0

        for x in xrange(x_min, x_max):
            film_x = -1.0+2.0*x/image_width
            for y in range(y_min, y_max):
                film_y = -1.0+2.0*y/image_height

                color = V3()
                fraction = 1.0/RAYS_PER_PIXEL
                for _ in xrange(RAYS_PER_PIXEL):

                    off_x = film_x + (random()*2-1)*pix_width
                    off_y = film_y + (random()*2-1)*pix_height

                    film_p = film_center + off_x*film_half_w*camera_x + \
                        off_y*film_half_h*camera_y

                    ray_origin = camera_pos
                    ray_dir = NoZ(film_p - camera_pos)

                    result = cast_ray(world, ray_origin, ray_dir)

                    color += result*fraction

                pixel = V3(LinearToRGB(color.x),
                           LinearToRGB(color.y),
                           LinearToRGB(color.z))
                pixel = Linear1ToRGB255(pixel)
                pixels[image_width - x - 1, y] = pixel.tuple()

        queue.bounces_computed += bounces_computed
        queue.tiles_rendered += 1
        print "\rRaycasting %1.2f %%...   " % \
            (100.0*queue.tiles_rendered/(queue.total_tile_count)),
        sys.stdout.flush()


def render(img, thread_count):

    world = World()

    world.materials.append(Material(V3(0.3, 0.4, 0.5), V3(0, 0, 0), 0.0))
    world.materials.append(Material(V3(0, 0, 0), V3(0.5, 0.5, 0.5), 0.0))
    world.materials.append(Material(V3(0, 0, 0), V3(0.7, 0.5, 0.3), 0.0))
    world.materials.append(Material(V3(2.0, 0.0, 0.0), V3(0, 0, 0), 0.0))
    world.materials.append(Material(V3(0, 0, 0), V3(0.2, 0.8, 0.2), 0.7))
    world.materials.append(Material(V3(0, 0, 0), V3(0.4, 0.8, 0.9), 0.85))
    world.materials.append(Material(V3(0, 0, 0), V3(0.95, 0.95, 0.95), 1.0))

    p = Plane(V3(0, 0, 1), 0, world.materials[1])
    world.planes.append(p)

    world.spheres.append(Sphere(V3(0, 0, 0), 1.0, world.materials[2]))
    world.spheres.append(Sphere(V3(3, -2, 0), 1.0, world.materials[3]))
    world.spheres.append(Sphere(V3(-2, -1, 2), 1.0, world.materials[4]))
    world.spheres.append(Sphere(V3(1, -1, 3), 1.0, world.materials[5]))
    world.spheres.append(Sphere(V3(-2, 3, 0), 2.0, world.materials[6]))

    start_time = time()

    # Set the tile width to be a power of two
    tile_width = 1
    while 2*tile_width < img.size[0]/thread_count:
        tile_width *= 2

    tile_height = tile_width
    tile_count_x = (img.size[0] + tile_width - 1) / tile_width
    tile_count_y = (img.size[1] + tile_width - 1) / tile_height

    print "Chunking: %d threads with %d %dx%d (%dk/tile) tiles" % \
        (thread_count, tile_count_x*tile_count_y, tile_width,
         tile_height, tile_width*tile_height*4/1024)

    queue = WorkQueue()
    queue.total_tile_count = tile_count_x * tile_count_y

    for tile_x in xrange(tile_count_x):
        min_x = tile_x*tile_width
        max_x = min(img.size[0], min_x + tile_width)
        for tile_y in xrange(tile_count_y):
            min_y = tile_y*tile_height
            max_y = min(img.size[1], min_y + tile_height)

            queue.work_orders.append((world, img, min_x, min_y, max_x, max_y))

    threads = list()
    for _ in xrange(thread_count):
        t = threading.Thread(target=render_tile, args=(queue,))
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    end_time = time()
    casting_time = (end_time - start_time)*1000

    print "\n"
    print "Raycasting time: %dms" % casting_time
    print "Total bounces: %d" % queue.bounces_computed
    print "Performance: %fms/bounce" % \
        (casting_time / (queue.bounces_computed+0.01))


def main():

    parser = argparse.ArgumentParser(description='A Simple Ray Tracer')
    parser.add_argument('-t', '--threads', type=int, default=1, nargs='?',
                        help='Number of threads to use')

    args = parser.parse_args()

    img = Image.new('RGB', (WIDTH, HEIGHT), "black")

    print "Rendering...",
    render(img, thread_count=args.threads)
    print "done."

    img.save('output.png')

    img.show()


if __name__ == "__main__":
    main()
