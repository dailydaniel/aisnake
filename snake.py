#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

from tkinter import *
import random
import time

from numpy import dot, arccos, clip
import numpy as np
from numpy.linalg import norm 
import math
import operator

from math import atan2
from math import cos
from math import degrees
from math import floor
from math import radians
from math import sin
from math import sqrt
from random import randint
# from random import random
from random import sample
from random import uniform

from ai import settings as settingsAI
from ai import dist, evolve, organism
 
import pickle

# Globals
settings = {}

settings['WIDTH'] = 800
settings['HEIGHT'] = 600
settings['SEG_SIZE'] = 20
settings['IN_GAME'] = True
settings['BLOCK'] = None
settings['TIME'] = 100
settings['TEXT'] = None
settings['CURRENT_ORG'] = None
settings['s'] = None
settings['c'] = None
settings['root'] = None


# Helper functions
def create_block(settings):
    """ Creates an apple to be eaten """
    posx = settings['SEG_SIZE'] * random.randint(1, (settings['WIDTH']-settings['SEG_SIZE']) / settings['SEG_SIZE'])
    posy = settings['SEG_SIZE'] * random.randint(1, (settings['HEIGHT']-settings['SEG_SIZE']) / settings['SEG_SIZE'])
    settings['BLOCK'] = settings['c'].create_oval(posx, posy,
                          posx+settings['SEG_SIZE'], posy+settings['SEG_SIZE'],
                          fill="red")


def gameover(settings):
    settings['c'].create_text(settings['WIDTH']/2, settings['HEIGHT']/2,
                  text="GAME OVER! score = {0}".format(settings['s'].score),
                  font="Arial 20",
                  fill="red")

def ang(v1, v2, r):
    u = np.array([1, 0]) 
    v = np.array([v2[0] - v1[0], v2[1] - v1[0]]) 
    c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle 
    angle = arccos(clip(c, -1, 1)) # if you really want the angle 
    angle *= 1 if v[1] >= 0 else -1
    angle /= math.pi
    angle -= r
    return angle

def centr(v):
    return ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2)

def smallcross(v1, v2):
    # v1 = [(0, 0), (1, 1)]
    # v2 = [(0, 1), (1, 0)]

    a = (v1[0][1] - v1[1][1], v2[0][1] - v2[1][1])
    b = (v1[1][0] - v1[0][0], v2[1][0] - v2[0][0])
    c = (v1[0][0] * v1[1][1] - v1[0][1] * v1[0][1], v2[0][0] * v2[1][1] - v2[0][1] * v2[0][1])

    if b[0] * a[1] - b[1] * a[0] and a[0]:
        y = (c[1] * a[0] - c[0] * a[1]) / (b[0] * a[1] - b[1] * a[0])
        x = (-c[0] - b[0] * y) / a[0]
        
        if min(v1[0][0], v1[1][0]) <= x <= max(v1[0][0], v1[1][0]):
            return True
    return False

def cross(settings, hc, bc):
    x, y = centr(hc), centr(bc)
    v1 = [(x[0], x[1]), (y[0], y[1])]

    for i in range(1, len(settings['s'].segments)-1):
        u1 = centr(settings['c'].coords(settings['s'].segments[i].instance))
        u2 = centr(settings['c'].coords(settings['s'].segments[i+1].instance))
        v2 = [(u1[0], u1[1]), (u2[0], u2[1])]
        if smallcross(v1, v2):
            return 1
    return 0

def dirs(settings, head_coords, back_coords, block_coords):
    r = settings['CURRENT_ORG'].r
    r_food = ang(centr(head_coords), centr(block_coords), r)
    r_back = ang(centr(head_coords), centr(back_coords), r)
    ifback = cross(settings, head_coords, block_coords)
    return r, r_food, r_back, ifback

def exvec(r):
    if 0.25 >= r >= -0.25:
        return "Right"
    elif 0.75 >= r >= 0.25:
        return "Up"
    elif -0.25 >= r >= -0.75:
        return "Down"
    # elif  r <= -0.75 or r >= 0.75:
    else:
        return "Left"

def h(r, head_coords, settings):
    h_c = centr(head_coords)
    real_r = exvec(r)

    if real_r == "Right":
        h = (settings['WIDTH'] - h_c[0]) / settings['WIDTH']
    elif real_r == "Up":
        h = h_c[1] / settings['HEIGHT']
    elif real_r == "Down":
        h = (settings['HEIGHT'] - h_c[1]) / settings['HEIGHT']
    else: # "Left"
        h = h_c[0] / settings['WIDTH']
    
    return h


def main():
    play(settings)

def play(settings):
    """ Handles game process """
    if settings['IN_GAME']:
        settings['s'].move(settings)
        head_coords = settings['c'].coords(settings['s'].segments[-1].instance)
        back_coords = settings['c'].coords(settings['s'].segments[0].instance)
        block_coords = settings['c'].coords(settings['BLOCK'])
        r, r_food, r_back, ifback = dirs(settings, head_coords, back_coords, block_coords)
        h_wall = h(r, head_coords, settings)

        x1, y1, x2, y2 = head_coords

        settings['CURRENT_ORG'].r_food = r_food
        settings['CURRENT_ORG'].r_back = r_back
        settings['CURRENT_ORG'].ifback = ifback
        settings['CURRENT_ORG'].h_wall = h_wall
        
        settings['CURRENT_ORG'].think()
        settings['CURRENT_ORG'].update_r()
        r = settings['CURRENT_ORG'].r
        settings['s'].change_direction_bot(exvec(r))

        settings['CURRENT_ORG'].fitness += 0.01 #
        

        # Check for collision with gamefield edges
        if x2 > settings['WIDTH'] or x1 < 0 or y1 < 0 or y2 > settings['HEIGHT']:
            settings['IN_GAME'] = False
            gameover(settings)

        # Eating apples
        elif head_coords == settings['c'].coords(settings['BLOCK']):
            settings['s'].score += 1
            settings['CURRENT_ORG'].fitness += 1

            settings['c'].delete(settings['TEXT'])
            settings['TEXT'] = settings['c'].create_text(50, settings['HEIGHT']-20, text="score = {0}".format(settings['s'].score))
            settings['s'].add_segment(settings)
            settings['c'].delete(settings['BLOCK'])
            create_block(settings)

        # Self-eating
        else:
            for index in range(len(settings['s'].segments)-1):
                if head_coords == settings['c'].coords(settings['s'].segments[index].instance):
                    settings['IN_GAME'] = False
                    gameover(settings)
        settings['root'].after(settings['TIME'], main)

    # Not IN_GAME -> stop game and print message
    else:
        time.sleep(1)
        settings['c'].delete(settings['s'])
        settings['root'].destroy()
        # settings['root'].quit()
        settings['IN_GAME'] = True


class Segment(object):
    """ Single snake segment """
    def __init__(self, settings, x, y):  # without c
        self.instance = settings['c'].create_rectangle(x, y,
                                           x+settings['SEG_SIZE'], y+settings['SEG_SIZE'],
                                           fill="white")
        self.energy = 1


class Snake(object):
    """ Simple Snake class """
    def __init__(self, segments):
        self.segments = segments
        # possible moves
        self.mapping = {"Down": (0, 1), "Right": (1, 0),
                        "Up": (0, -1), "Left": (-1, 0)}
        # initial movement direction
        self.vector = self.mapping["Right"]
        self.score = 0

    def move(self, settings):
        """ Moves the snake with the specified vector"""
        for index in range(len(self.segments)-1):
            segment = self.segments[index].instance
            x1, y1, x2, y2 = settings['c'].coords(self.segments[index+1].instance)
            settings['c'].coords(segment, x1, y1, x2, y2)

        x1, y1, x2, y2 = settings['c'].coords(self.segments[-2].instance)
        settings['c'].coords(self.segments[-1].instance,
                 x1+self.vector[0]*settings['SEG_SIZE'], y1+self.vector[1]*settings['SEG_SIZE'],
                 x2+self.vector[0]*settings['SEG_SIZE'], y2+self.vector[1]*settings['SEG_SIZE'])

    def add_segment(self, settings):
        """ Adds segment to the snake """
        last_seg = settings['c'].coords(self.segments[0].instance)
        x = last_seg[2] - settings['SEG_SIZE']
        y = last_seg[3] - settings['SEG_SIZE']
        self.segments.insert(0, Segment(settings, x, y))

    def change_direction(self, event):
        """ Changes direction of snake """
        if event.keysym in self.mapping:
            self.vector = self.mapping[event.keysym]
    def change_direction_bot(self, event):

        """ Changes direction of snake by ai """
        if event in self.mapping:
            self.vector = self.mapping[event]


def simulate(settings, settingsAI, organisms, gen):
    new_organisms = []
    for org in organisms:
        settings['CURRENT_ORG'] = org

        settings['root'] = Tk()
        settings['root'].title("Snake")

        settings['c'] = Canvas(settings['root'], width=settings['WIDTH'], height=settings['HEIGHT'], bg="#003300")
        settings['c'].grid()

        segments = [Segment(settings, settings['SEG_SIZE'], settings['SEG_SIZE']),
                    Segment(settings, settings['SEG_SIZE']*2, settings['SEG_SIZE']),
                    Segment(settings, settings['SEG_SIZE']*3, settings['SEG_SIZE'])]
        settings['s'] = Snake(segments)
        # settings['s'] = s

        settings['c'].focus_set()
        settings['c'].bind("<KeyPress>", settings['s'].change_direction)
        create_block(settings)
        settings['TEXT'] = settings['c'].create_text(50, settings['HEIGHT']-20, text="score = 0")
        
        main()
        settings['root'].mainloop()
        
        new_organisms.append(settings['CURRENT_ORG'])
    return new_organisms


# simulate(settings)

def run(settings, settingsAI):
    organisms = []
    for i in range(settingsAI['pop_size']):
        wih_init = np.random.uniform(-1, 1, (settingsAI['hnodes'], settingsAI['inodes']))     # mlp weights (input -> hidden)
        who_init = np.random.uniform(-1, 1, (settingsAI['onodes'], settingsAI['hnodes']))     # mlp weights (hidden -> output)

        organisms.append(organism(settingsAI, wih_init, who_init, name='gen[x]-org['+str(i)+']'))

    for gen in range(settingsAI['gens']):
        organisms = simulate(settings, settingsAI, organisms, gen)
        organisms, stats = evolve(settingsAI, organisms, gen)
        print('> GEN:',gen,'BEST:',stats['BEST'],'AVG:',stats['AVG'],'WORST:',stats['WORST'])

    with open('org.pickle.dat', 'wb') as f:
        pickle.dump(organisms)    


run(settings, settingsAI)
