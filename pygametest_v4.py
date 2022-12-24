# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:00:01 2022

@author: lukej
"""

import pygame
import numpy as np

# notes: 
# add penalty for interacting
# create Dot class which also has attributes colour, value

GRID_SZ = 600
GRID_RAD = GRID_SZ/2
SEG_ANGLE = np.pi/6
SPHERE_RAD = GRID_RAD/SEG_ANGLE
BACKGRND_CLR = (0,0,0)
FOREGRND_CLR = (255,255,255)
FPS = 60
VEL = 3
SIGMA = 0.5

P_DOT_NO = 3
P_COLORS = [[0, 100, 200], [100, 0, 200], [0,200,100]]
# r, g, b = P_DOT_CLR[0]
# print(r)
P_VALUE = 100
COLLISION_P = pygame.USEREVENT + 1
N_DOT_NO = 1
N_COLORS = [[200,0,0]]
DOT_SZ = 10
N_VALUE = -100
COLLISION_N = pygame.USEREVENT + 2

AGENT_SZ = 12
AGENT_CLR = (50,50,50)
INTERACT = pygame.USEREVENT + 3

N = 10
RF_RAD = 0.5
MARK_SZ = 0.5
N_TCKS = 5

pygame.font.init()
FONT = pygame.font.SysFont('arial', 22)
WIN = pygame.display.set_mode((GRID_SZ, GRID_SZ))
pygame.display.set_caption('sim_v4')

def create_dot(grid_rad, dot_sz, value):
    r = np.random.uniform(0, (grid_rad-DOT_SZ/2)**2) ** 0.5
    t = np.random.uniform(0, 2*np.pi)
    x = grid_rad + r*np.cos(t)
    y = grid_rad + r*np.sin(t)
    d = pygame.Rect(x, y, dot_sz, dot_sz)
    # dot.value = value
    # dot.colour = colour
    return(d)

def generate_p_dots(p_dot_no, grid_rad, dot_sz, p_value):
    p_dots = {}
    for i in range(p_dot_no):
        dot_i = "pdot_%d" % i
        p_dots[dot_i] = create_dot(grid_rad, dot_sz, p_value)
    return(p_dots)

def generate_n_dots(n_dot_no, grid_rad, dot_sz, n_value):
    n_dots = {}
    for i in range(n_dot_no):
        dot_i = "ndot_%d" % i
        n_dots[dot_i] = create_dot(grid_rad, dot_sz, n_value)
    return(n_dots)
    
def draw_window(agent, p_dots, n_dots, score, iteration):
    WIN.fill(BACKGRND_CLR)
    pygame.draw.circle(WIN, FOREGRND_CLR, (GRID_SZ/2, GRID_SZ/2), GRID_RAD)
    pygame.draw.rect(WIN, AGENT_CLR, agent)
    i = 0
    for dot in p_dots.values():
        pygame.draw.circle(WIN, P_COLORS[i], (dot.left + dot.width/2, dot.top + dot.height/2), dot.width/2)
        i += 1
    i = 0
    for dot in n_dots.values():
        pygame.draw.circle(WIN, N_COLORS[i], (dot.left + dot.width/2, dot.top + dot.height/2), dot.width/2)
        i += 1
        
    score_text = FONT.render('Score: ' + str(score), 1, FOREGRND_CLR)
    iteration_text = FONT.render('Iteration: ' + str(iteration), 1, FOREGRND_CLR)
    WIN.blit(score_text, (GRID_SZ - score_text.get_width() - 5, 5))
    WIN.blit(iteration_text, (GRID_SZ - iteration_text.get_width() - 5, 30))
    
    pygame.display.update()

def dots_move(keys_pressed,p_dots,n_dots):
    if keys_pressed[pygame.K_LEFT]: 
            for dot in p_dots.values():
                dot.x += VEL
            for dot in n_dots.values():
                dot.x += VEL
    if keys_pressed[pygame.K_RIGHT]: 
            for dot in p_dots.values():
                dot.x -= VEL
            for dot in n_dots.values():
                dot.x -= VEL    
    if keys_pressed[pygame.K_UP]: 
            for dot in p_dots.values():
                dot.y += VEL
            for dot in n_dots.values():
                dot.y += VEL     
    if keys_pressed[pygame.K_DOWN]: 
            for dot in p_dots.values():
                dot.y -= VEL
            for dot in n_dots.values():
                dot.y -= VEL                 
            
def handle_collision_p(agent,p_dots,keys_pressed,score):
    for dot in p_dots.values():
        if agent.colliderect(dot) and keys_pressed[pygame.K_SPACE]:
            pygame.event.post(pygame.event.Event(COLLISION_P))
        x = GRID_RAD + (GRID_RAD/np.sin(SEG_ANGLE))*np.sin(((dot.x+dot.width/2)-GRID_RAD)*(SEG_ANGLE/GRID_RAD))
        y = GRID_RAD - (GRID_RAD/np.sin(SEG_ANGLE))*np.sin((GRID_RAD-(dot.y+dot.height/2))*(SEG_ANGLE/GRID_RAD))
        if ((x-GRID_RAD)**2 + (y-GRID_RAD)**2) >= GRID_RAD**2:
            dot.height = 0
            dot.width = 0
        else:
            dot.height = DOT_SZ
            dot.width = DOT_SZ

def handle_collision_n(agent,n_dots,keys_pressed,score):
    for dot in n_dots.values():
        if agent.colliderect(dot) and keys_pressed[pygame.K_SPACE]:
            pygame.event.post(pygame.event.Event(COLLISION_N))
        x = GRID_RAD + (GRID_RAD/np.sin(SEG_ANGLE))*np.sin(((dot.x+dot.width/2)-GRID_RAD)*(SEG_ANGLE/GRID_RAD))
        y = GRID_RAD - (GRID_RAD/np.sin(SEG_ANGLE))*np.sin((GRID_RAD-(dot.y+dot.height/2))*(SEG_ANGLE/GRID_RAD))
        if ((x-GRID_RAD)**2 + (y-GRID_RAD)**2) >= GRID_RAD**2:
            dot.height = 0
            dot.width = 0
        else:
            dot.height = DOT_SZ
            dot.width = DOT_SZ
            
def create_arr_i(N):
    i = np.linspace(0,N-1,N)
    th_i = np.reshape((2*np.pi/N)*i,(1,i.size))
    arr_i = np.tile(np.array(th_i).transpose(),(1,N)) # varies in y
    return arr_i

def create_arr_j(N):
    j = np.linspace(0,N-1,N)
    th_j = np.reshape((2*np.pi/N)*j,(1,j.size))
    arr_j = np.tile(np.array(th_j),(N,1)) # varies in x       
    return arr_j
            
#call r_test on all dots, neuron array
def r_test(p_dots,n_dots,p_colors,n_colors,arr_j,arr_i,sigma):
    r_arr = np.zeros([arr_i.size,1])
    i = 0
    for dot in p_dots.values():
        r,g,b = p_colors[i]
        r_arr += r/255 * f(dot,arr_j,arr_i,sigma)
        i += 1
    i = 0
    for dot in n_dots.values():
        r,g,b = n_colors[i]
        r_arr += r/255 * f(dot,arr_j,arr_i,sigma)
        i += 1
    return r_arr

# whole array; after debugging r_test
def arr_neuron(p_dots,n_dots,p_colors,n_colors,arr_j,arr_i,sigma):
    arr_n = np.zeros([arr_i.size,3])
    i = 0
    for dot in p_dots.values():
        r,g,b = p_colors[i]
        f_ = f(dot,arr_j,arr_i,sigma)
        arr_n[:,0] += r/255 * f_
        arr_n[:,1] += g/255 * f_
        arr_n[:,2] += b/255 * f_
        i += 1
    i = 0
    for dot in n_dots.values():
        r,g,b = n_colors[i]
        f_ = f(dot,arr_j,arr_i,sigma)
        arr_n[:,0] += r/255 * f_
        arr_n[:,1] += g/255 * f_
        arr_n[:,2] += b/255 * f_
        i += 1
    return arr_n

#call f on particular dot, returning values to each corresponding neuron in array
def f(dot,arr_j,arr_i,sigma):
    th_1 = (dot.x-GRID_RAD)*(SEG_ANGLE/GRID_RAD)
    #(np.pi/GRID_RAD)/SPHERE_RAD
    th_2 = (GRID_RAD-dot.y)*(SEG_ANGLE/GRID_RAD)
    #(np.pi/GRID_RAD)/SPHERE_RAD
    del_1 = th_1*np.ones([1,N]) - np.array(arr_j[1,:])
    del_2 = th_2*np.ones([N,1]) - np.array(arr_i[:,1]).reshape(N,1)
    # print(del_1.size)
    # print(del_2.size)
    # need to edit; should be N x N; corresponding to each neuron
    f_ = np.exp((np.cos(np.tile(del_1,(N,1))) + np.cos(np.tile(del_2,(1,N)) - 2))/sigma**2)
    f_ = (f_.T).ravel().reshape([f_.size,1])
    # print('size is',f_.size)
    return f_

def main():
    score = 0
    iteration = 0

    ARR_i = create_arr_i(N)
    ARR_j = create_arr_j(N)
    arr_N = np.zeros([ARR_i.size,3])

    agent = pygame.Rect(GRID_SZ/2-AGENT_SZ/2, GRID_SZ/2-AGENT_SZ/2, AGENT_SZ, AGENT_SZ)

    p_dots = generate_p_dots(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
    n_dots = generate_n_dots(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
    
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        
        keys_pressed = pygame.key.get_pressed()
        dots_move(keys_pressed, p_dots, n_dots)
        handle_collision_p(agent, p_dots, keys_pressed, score)
        handle_collision_n(agent, n_dots, keys_pressed, score)
        
        for event in pygame.event.get():
            if event.type == COLLISION_P:
                score += P_VALUE
                iteration += 1
                agent.left = GRID_SZ/2
                agent.top = GRID_SZ/2
                p_dots = generate_p_dots(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
                n_dots = generate_n_dots(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
           
            if event.type == COLLISION_N:
                score += N_VALUE
                iteration += 1
                agent.left = GRID_SZ/2
                agent.top = GRID_SZ/2
                p_dots = generate_p_dots(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
                n_dots = generate_n_dots(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
                
            if event.type == pygame.QUIT:
                run = False
        
        #arr_N = arr_neuron(p_dots,n_dots,P_COLORS,N_COLORS,ARR_j,ARR_i,SIGMA)
        #print(arr_N)
        r_arr = r_test(p_dots,n_dots,P_COLORS,N_COLORS,ARR_j,ARR_i,SIGMA)
        print(r_arr)    
        
        draw_window(agent, p_dots, n_dots, score, iteration)

    pygame.quit()

if __name__ == "__main__":
    main()
