# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:09:18 2022

@author: lukej
"""

import pygame
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.patches import Circle
#import pytorch

### global parameters
GRID_SZ = 600
GRID_RAD = GRID_SZ/2
BACKGRND_CLR = (0,0,0)
FOREGRND_CLR = (255,255,255)
FPS = 60
VEL = 3

### neuron parameters
N = 10 # neurons on each axis
APERTURE = np.pi/3 # aperture size
SIGMA = 1 # receptive field size
SPHERE_RAD = GRID_RAD/APERTURE

### dot parameters
DOT_SZ = 8 # dot size
P_DOT_NO = 3 # number of extra dots
P_COLORS = [[0, 100, 200], [100, 0, 200], [0,200,100]]
P_REWARD = 0
COLLISION_P = pygame.USEREVENT + 1
N_DOT_NO = 1 # number of desired dots
N_COLORS = [[200,0,0]]
N_REWARD = 100
COLLISION_N = pygame.USEREVENT + 2

### agent parameters
AGENT_SZ = 12
AGENT_CLR = (50,50,50)
INTERACT = pygame.USEREVENT + 3

pygame.font.init()
FONT = pygame.font.SysFont('arial', 19)
WIN = pygame.display.set_mode((GRID_SZ, GRID_SZ))
pygame.display.set_caption('sim_v7')

def neurons_j(N,APERTURE):
    th_j = np.reshape(np.linspace(0,2*APERTURE,N),(1,N))
    arr_j = np.tile(np.array(th_j),(N,1)) # varies in x
    return arr_j
    
def neurons_i(N,APERTURE):
    th_i = np.reshape(np.linspace(0,2*APERTURE,N),(N,1))
    arr_i = np.tile(np.array(th_i),(1,N)) # varies in y
    return arr_i

# def neuron_arr(N,APERTURE):
#     th = np.reshape(np.linspace(0,2*APERTURE,N),(1,N))
#     arr = np.tile(np.array(th),(N,1))

def create_dot(grid_sz, dot_sz, value):
    d_j = (grid_sz/(2*np.pi))*np.random.uniform(0,2*np.pi)
    d_i = (grid_sz/(2*np.pi))*np.random.uniform(0,2*np.pi)
    d = pygame.Rect(d_j, d_i, dot_sz, dot_sz)
    return d

def generate_p_dots(p_dot_no, grid_sz, dot_sz, p_reward):
    p_dots = {}
    for i in range(p_dot_no):
        dot_i = "pdot_%d" % i
        p_dots[dot_i] = create_dot(grid_sz, dot_sz, p_reward)
    return(p_dots)

def generate_n_dots(n_dot_no, grid_sz, dot_sz, n_reward):
    n_dots = {}
    for i in range(n_dot_no):
        dot_i = "ndot_%d" % i
        n_dots[dot_i] = create_dot(grid_sz, dot_sz, n_reward)
    return(n_dots)
    
def draw_window(agent, p_dots, n_dots, score, iteration, grid_sz, NEUR_i, NEUR_j):
    WIN.fill(FOREGRND_CLR)
    for i in range(len(NEUR_i)):
        for j in range(len(NEUR_j)):
            n = pygame.Rect(GRID_SZ/(2*np.pi)*(np.pi-APERTURE+NEUR_i[i,0]),GRID_SZ/(2*np.pi)*(np.pi-APERTURE+NEUR_j[0,j]),DOT_SZ/8,DOT_SZ/8)
            pygame.draw.rect(WIN,BACKGRND_CLR,n)
    pygame.draw.rect(WIN, AGENT_CLR, agent)
    i = 0
    for dot in p_dots.values():
        pygame.draw.circle(WIN, P_COLORS[i], (dot.left+dot.width/2,dot.top+dot.width/2), dot.width/2)
        i += 1
    i = 0
    for dot in n_dots.values():
        pygame.draw.circle(WIN, N_COLORS[i], (dot.left+dot.width/2,dot.top+dot.width/2), dot.width/2)
        i += 1
        
    score_text = FONT.render('Score: ' + str(score), 1, BACKGRND_CLR)
    iteration_text = FONT.render('Iteration: ' + str(iteration), 1, BACKGRND_CLR)
    WIN.blit(score_text, (GRID_SZ - score_text.get_width() - 5, 5))
    WIN.blit(iteration_text, (GRID_SZ - iteration_text.get_width() - 5, 30))
    
    pygame.display.update()

def dots_move(keys_pressed,p_dots,n_dots):
    v = np.zeros([2,1])
    if keys_pressed[pygame.K_LEFT]:
            for dot in p_dots.values():
                dot.x += VEL
            for dot in n_dots.values():
                dot.x += VEL
            v[0] = -VEL
    if keys_pressed[pygame.K_RIGHT]: 
            for dot in p_dots.values():
                dot.x -= VEL
            for dot in n_dots.values():
                dot.x -= VEL    
            v[0] = VEL
    if keys_pressed[pygame.K_UP]: 
            for dot in p_dots.values():
                dot.y += VEL
            for dot in n_dots.values():
                dot.y += VEL     
            v[1] = VEL
    if keys_pressed[pygame.K_DOWN]: 
            for dot in p_dots.values():
                dot.y -= VEL
            for dot in n_dots.values():
                dot.y -= VEL     
            v[1] = -VEL
    return v
            
def handle_collision_p(agent,p_dots,keys_pressed,score):
    for dot in p_dots.values():
        if agent.colliderect(dot):
            pygame.event.post(pygame.event.Event(COLLISION_P))
            dot.height = 0
            dot.width = 0
        if dot.x > GRID_SZ:
            dot.x = 0
        if dot.y > GRID_SZ:
            dot.y = 0
        if dot.x < 0:
            dot.x = GRID_SZ
        if dot.y < 0:
            dot.y = GRID_SZ

def handle_collision_n(agent,n_dots,keys_pressed,score):
    for dot in n_dots.values():
        if agent.colliderect(dot):
            pygame.event.post(pygame.event.Event(COLLISION_N))
        if dot.x > GRID_SZ:
            dot.x = 0
        if dot.y > GRID_SZ:
            dot.y = 0
        if dot.x < 0:
            dot.x = GRID_SZ
        if dot.y < 0:
            dot.y = GRID_SZ

def act_neuron(p_dots,n_dots,p_colors,n_colors,arr_j,arr_i,sigma):
    act_r = act_g = act_b = np.zeros([arr_i.size,1])
    i = 0
    for dot in p_dots.values():
        r,g,b = p_colors[i]
        f_ = f(dot,arr_j,arr_i,sigma)
        act_r += r/255 * f_.reshape([N*N,1])
        act_g += g/255 * f_.reshape([N*N,1])
        act_b += b/255 * f_.reshape([N*N,1])
        i += 1
    i = 0
    for dot in n_dots.values():
        r,g,b = n_colors[i]
        f_ = f(dot,arr_j,arr_i,sigma)
        act_r += r/255 * f_.reshape([N*N,1])
        act_g += g/255 * f_.reshape([N*N,1])
        act_b += b/255 * f_.reshape([N*N,1])
        i += 1
    act_n = np.concatenate((act_r,act_g,act_b),axis=1)
    return act_n

def f(dot,arr_j,arr_i,sigma):
    del_j = np.ones([1,N])*dot.x - arr_j[0,:]
    del_i = np.ones([N,1])*dot.y - np.array(arr_i[:,0]).reshape(N,1)
    f_ = np.exp(((np.cos(np.tile(del_j,(N,1))) + np.cos(np.tile(del_i,(1,N))) - 2))/sigma**2)
    f_ = f_.ravel().reshape([f_.size,1])
    return f_

def main():
    score = 0
    iteration = 0

    NEUR_i = neurons_i(N,APERTURE)
    NEUR_j = neurons_j(N,APERTURE)
    activations = np.zeros([NEUR_i.size,3])

    agent = pygame.Rect(GRID_RAD-AGENT_SZ/2, GRID_RAD-AGENT_SZ/2, AGENT_SZ, AGENT_SZ)

    p_dots = generate_p_dots(P_DOT_NO, GRID_SZ, DOT_SZ, P_REWARD)
    n_dots = generate_n_dots(N_DOT_NO, GRID_SZ, DOT_SZ, N_REWARD)
    
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        
        keys_pressed = pygame.key.get_pressed()
        v = dots_move(keys_pressed, p_dots, n_dots)
        print(v)
        handle_collision_p(agent, p_dots, keys_pressed, score)
        handle_collision_n(agent, n_dots, keys_pressed, score)
        
        for event in pygame.event.get():
            if event.type == COLLISION_P:
                score += P_REWARD
                # iteration += 1
                # agent.left = GRID_RAD-AGENT_SZ/2
                # agent.top = GRID_RAD-AGENT_SZ/2
                # p_dots = generate_p_dots(P_DOT_NO, GRID_RAD, DOT_SZ, P_REWARD)
                # n_dots = generate_n_dots(N_DOT_NO, GRID_RAD, DOT_SZ, N_REWARD)
           
            if event.type == COLLISION_N:
                score += N_REWARD
                iteration += 1
                agent.left = GRID_RAD-AGENT_SZ/2
                agent.top = GRID_RAD-AGENT_SZ/2
                p_dots = generate_p_dots(P_DOT_NO, GRID_RAD, DOT_SZ, P_REWARD)
                n_dots = generate_n_dots(N_DOT_NO, GRID_RAD, DOT_SZ, N_REWARD)
                
            if event.type == pygame.QUIT:
                run = False
        
        activations = act_neuron(p_dots,n_dots,P_COLORS,N_COLORS,NEUR_j,NEUR_i,SIGMA)
        print(activations)
        
        draw_window(agent, p_dots, n_dots, score, iteration, GRID_SZ,NEUR_i,NEUR_j)

    pygame.quit()

if __name__ == "__main__":
    main()