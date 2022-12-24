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
SPHERE_RAD = GRID_RAD
BACKGRND_CLR = (0,0,0)
FOREGRND_CLR = (255,255,255)
FPS = 60
VEL = 3

P_DOT_NO = 3
P_DOT_CLR = [[0, 100, 200], [100, 0, 200], [0,200,100]]
P_VALUE = 100
COLLISION_P = pygame.USEREVENT + 1
N_DOT_NO = 1
N_DOT_CLR = ['red']
DOT_SZ = 10
N_VALUE = -100
COLLISION_N = pygame.USEREVENT + 2

AGENT_SZ = 12
AGENT_CLR = (50,50,50)
INTERACT = pygame.USEREVENT + 3

pygame.font.init()
FONT = pygame.font.SysFont('arial', 22)
WIN = pygame.display.set_mode((GRID_SZ, GRID_SZ))
pygame.display.set_caption('sim_v2')

def create_dot(grid_rad, dot_sz, value):
    r = np.random.uniform(0, (grid_rad-DOT_SZ/2)**2) ** 0.5
    t = np.random.uniform(0, 2*np.pi)
    x = grid_rad + r*np.cos(t)
    y = grid_rad + r*np.sin(t)
    d = pygame.Rect(x, y, dot_sz, dot_sz)
    # dot.value = value
    # dot.colour = colour
    return(d)

def generate_dots_p(p_dot_no, grid_rad, dot_sz, p_value):
    dots_p = {}
    for i in range(p_dot_no):
        dot_i = "pdot_%d" % i
        dots_p[dot_i] = create_dot(grid_rad, dot_sz, p_value)
    return(dots_p)

def generate_dots_n(n_dot_no, grid_rad, dot_sz, n_value):
    dots_n = {}
    for i in range(n_dot_no):
        dot_i = "ndot_%d" % i
        dots_n[dot_i] = create_dot(grid_rad, dot_sz, n_value)
    return(dots_n)
    
def draw_window(agent, dots_p, dots_n, score, iteration):
    WIN.fill(BACKGRND_CLR)
    pygame.draw.circle(WIN, FOREGRND_CLR, (GRID_SZ/2, GRID_SZ/2), GRID_RAD)
    pygame.draw.rect(WIN, AGENT_CLR, agent)
    i = 0
    for dot in dots_p.values():
        pygame.draw.circle(WIN, P_DOT_CLR[i], (dot.left + dot.width/2, dot.top + dot.height/2), dot.width/2)
        i += 1
    i = 0
    for dot in dots_n.values():
        pygame.draw.circle(WIN, N_DOT_CLR[i], (dot.left + dot.width/2, dot.top + dot.height/2), dot.width/2)
        i += 1
        
    score_text = FONT.render('Score: ' + str(score), 1, FOREGRND_CLR)
    iteration_text = FONT.render('Iteration: ' + str(iteration), 1, FOREGRND_CLR)
    WIN.blit(score_text, (GRID_SZ - score_text.get_width() - 5, 5))
    WIN.blit(iteration_text, (GRID_SZ - iteration_text.get_width() - 5, 30))
    
    pygame.display.update()

def dots_move(keys_pressed,dots_p,dots_n):
    if keys_pressed[pygame.K_LEFT]: 
            for dot in dots_p.values():
                dot.x += VEL
            for dot in dots_n.values():
                dot.x += VEL
    if keys_pressed[pygame.K_RIGHT]: 
            for dot in dots_p.values():
                dot.x -= VEL
            for dot in dots_n.values():
                dot.x -= VEL    
    if keys_pressed[pygame.K_UP]: 
            for dot in dots_p.values():
                dot.y += VEL
            for dot in dots_n.values():
                dot.y += VEL     
    if keys_pressed[pygame.K_DOWN]: 
            for dot in dots_p.values():
                dot.y -= VEL
            for dot in dots_n.values():
                dot.y -= VEL                 
            
def handle_collision_p(agent,dots_p,keys_pressed,score):
    for dot in dots_p.values():
        if agent.colliderect(dot) and keys_pressed[pygame.K_SPACE]:
            pygame.event.post(pygame.event.Event(COLLISION_P))
        if ((dot.x-GRID_RAD)**2 + (dot.y-GRID_RAD)**2) >= GRID_RAD**2:
            dot.height = 0
            dot.width = 0
        else:
            dot.height = DOT_SZ
            dot.width = DOT_SZ

def handle_collision_n(agent,dots_n,keys_pressed,score):
    for dot in dots_n.values():
        if agent.colliderect(dot) and keys_pressed[pygame.K_SPACE]:
            pygame.event.post(pygame.event.Event(COLLISION_N))
        if ((dot.x-GRID_RAD)**2 + (dot.y-GRID_RAD)**2) >= GRID_RAD**2:
            dot.height = 0
            dot.width = 0
        else:
            dot.height = DOT_SZ
            dot.width = DOT_SZ
            
# create 2d grid of polar coords
## define reeceptive field for single point
## define points corresponding to each neurons
## use g's formula to detremine firing (intensity) of neuron

def main():
    score = 0
    iteration = 0

    agent = pygame.Rect(GRID_SZ/2-AGENT_SZ/2, GRID_SZ/2-AGENT_SZ/2, AGENT_SZ, AGENT_SZ)

    dots_p = generate_dots_p(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
    dots_n = generate_dots_n(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
    
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        
        keys_pressed = pygame.key.get_pressed()
        dots_move(keys_pressed, dots_p, dots_n)
        handle_collision_p(agent, dots_p, keys_pressed, score)
        handle_collision_n(agent, dots_n, keys_pressed, score)
        
        for event in pygame.event.get():
            if event.type == COLLISION_P:
                score += P_VALUE
                iteration += 1
                agent.left = GRID_SZ/2
                agent.top = GRID_SZ/2
                dots_p = generate_dots_p(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
                dots_n = generate_dots_n(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
           
            if event.type == COLLISION_N:
                score += N_VALUE
                iteration += 1
                agent.left = GRID_SZ/2
                agent.top = GRID_SZ/2
                dots_p = generate_dots_p(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
                dots_n = generate_dots_n(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
                
            if event.type == pygame.QUIT:
                run = False

        draw_window(agent, dots_p, dots_n, score, iteration)

    pygame.quit()

if __name__ == "__main__":
    main()
