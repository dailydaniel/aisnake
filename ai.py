#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division
from collections import defaultdict

import numpy as np
import operator

from math import atan2
from math import cos
from math import degrees
from math import floor
from math import radians
from math import sin
from math import sqrt
from random import randint
from random import random
from random import sample
from random import uniform


#--- CONSTANTS ----------------------------------------------------------------+


settings = {}

# EVOLUTION SETTINGS 
settings['pop_size'] = 50        # number of organisms       50
settings['gens'] = 50            # number of generations     50
settings['elitism'] = 0.20      # elitism (selection bias)
settings['mutate'] = 0.10       # mutation rate

# SIMULATION SETTINGS
# settings['x_min'] = -2.0        # arena western border
# settings['x_max'] =  2.0        # arena eastern border
# settings['y_min'] = -2.0        # arena southern border
# settings['y_max'] =  2.0        # arena northern border

# ORGANISM NEURAL NET SETTINGS
settings['inodes'] = 4          # number of input nodes 
settings['hnodes'] = 5          # number of hidden nodes
settings['onodes'] = 1          # number of output nodes


#--- FUNCTIONS ----------------------------------------------------------------+


def dist(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)


def evolve(settings, organisms_old, gen):

    elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_orgs = settings['pop_size'] - elitism_num

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness
            
        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']
    
    
    #--- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = []
    for i in range(0, elitism_num):
        organisms_new.append(organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name))

    
    #--- GENERATE NEW ORGANISMS ---------------------------+
    for w in range(0, new_orgs):

        # SELECTION (TRUNCATION SELECTION)
        canidates = range(0, elitism_num)
        random_index = sample(canidates, 2)
        org_1 = orgs_sorted[random_index[0]]
        org_2 = orgs_sorted[random_index[1]]

        # CROSSOVER
        crossover_weight = random()
        wih_new = (crossover_weight * org_1.wih) + ((1 - crossover_weight) * org_2.wih)
        who_new = (crossover_weight * org_1.who) + ((1 - crossover_weight) * org_2.who)
        
        # MUTATION
        mutate = random()
        if mutate <= settings['mutate']:

            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0,1)     

            # MUTATE: WIH WEIGHTS
            if mat_pick == 0:
                index_row = randint(0,settings['hnodes']-1)
                index_col = randint(0,settings['inodes']-1)
                wih_new[index_row][index_col] = wih_new[index_row][index_col] * uniform(0.9, 1.1)
                if wih_new[index_row][index_col] >  1: wih_new[index_row][index_col] = 1
                if wih_new[index_row][index_col] < -1: wih_new[index_row][index_col] = -1
                
            # MUTATE: WHO WEIGHTS
            if mat_pick == 1:
                index_row = randint(0,settings['onodes']-1)
                index_col = randint(0,settings['hnodes']-1)
                who_new[index_row][index_col] = who_new[index_row][index_col] * uniform(0.9, 1.1)
                if who_new[index_row][index_col] >  1: who_new[index_row][index_col] = 1
                if who_new[index_row][index_col] < -1: who_new[index_row][index_col] = -1
                    
        organisms_new.append(organism(settings, wih=wih_new, who=who_new, name='gen['+str(gen)+']-org['+str(w)+']'))
                
    return organisms_new, stats


#--- CLASSES ------------------------------------------------------------------+


class organism():
    def __init__(self, settings, wih=None, who=None, name=None):

        self.r = 0          # orientation   [0, 360]

        self.r_food = 0     # orientation to nearest food
        self.r_back = 1.0   # orientation to back
        self.ifback = 0     # 0 or 1 if back on the way to food
        self.fitness = 0    # fitness (food count)
        self.h_wall = 1     # normalized distance to wall by orintation

        self.wih = wih
        self.who = who

        self.name = name
        
        
    # NEURAL NETWORK
    def think(self):

        # SIMPLE MLP
        af = lambda x: np.tanh(x)               # activation function
        h1 = af(np.dot(self.wih, [self.r_food, self.r_back, self.ifback, self.h_wall]))  # hidden layer
        out = af(np.dot(self.who, h1))          # output layer

        # UPDATE dv AND dr WITH MLP RESPONSE
        self.nn_dr = float(out[0])   # [-1, 1]  (up=1, down=-1) 

        
    # UPDATE HEADING
    def update_r(self):
        self.r += self.nn_dr 
        self.r %= 1 if self.r >= 0 else -1


#--- MAIN ---------------------------------------------------------------------+


# def run(settings):

#     #--- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
#     print('pt 1')
#     foods = []
#     for i in range(0,settings['food_num']):
#         foods.append(food(settings))

#     #--- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
#     print('pt 2')
#     organisms = []
#     for i in range(0,settings['pop_size']):
#         wih_init = np.random.uniform(-1, 1, (settings['hnodes'], settings['inodes']))     # mlp weights (input -> hidden)
#         who_init = np.random.uniform(-1, 1, (settings['onodes'], settings['hnodes']))     # mlp weights (hidden -> output)
        
#         organisms.append(organism(settings, wih_init, who_init, name='gen[x]-org['+str(i)+']'))
    
#     #--- CYCLE THROUGH EACH GENERATION --------------------+
#     print('pt 3')
#     for gen in range(0, settings['gens']):
        
#         # SIMULATE
#         organisms = simulate(settings, organisms, foods, gen)

#         # EVOLVE
#         organisms, stats = evolve(settings, organisms, gen)
#         print('> GEN:',gen,'BEST:',stats['BEST'],'AVG:',stats['AVG'],'WORST:',stats['WORST'])

#     pass


#--- RUN ----------------------------------------------------------------------+

#run(settings)
    
#--- END ----------------------------------------------------------------------+