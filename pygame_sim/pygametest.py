import pygame
import numpy as np

# notes: 
# add grid lines
# create Dot class which also has attributes colour, value

GRID_SZ = 600
GRID_RAD = GRID_SZ/2
BACKGRND_CLR = (0,0,0)
FOREGRND_CLR = (255,255,255)
FPS = 30
VEL = 5

P_DOT_NO = 3
P_DOT_CLR = ['magenta', 'blue', 'green']
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
SCORE_FONT = pygame.font.SysFont('arial',30)
WIN = pygame.display.set_mode((GRID_SZ, GRID_SZ))
pygame.display.set_caption('sim_v1')

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
    
def draw_window(agent, dots_p, dots_n, score):
    WIN.fill(BACKGRND_CLR)
    pygame.draw.circle(WIN, FOREGRND_CLR, (GRID_SZ/2, GRID_SZ/2), GRID_RAD)
    pygame.draw.rect(WIN, AGENT_CLR, agent)
    i = 0
    for key in dots_p:
        pygame.draw.circle(WIN, P_DOT_CLR[i], (dots_p[key].left + dots_p[key].width/2, dots_p[key].top + dots_p[key].height/2), dots_p[key].width/2)
        i += 1
    i = 0
    for key in dots_n:
        pygame.draw.circle(WIN, N_DOT_CLR[i], (dots_n[key].left + dots_n[key].width/2, dots_n[key].top + dots_n[key].height/2), dots_n[key].width/2)
        i += 1
        
    score_text = SCORE_FONT.render('Score: ' + str(score), 1, FOREGRND_CLR)
    WIN.blit(score_text, (GRID_SZ - score_text.get_width() - 5, 5))
    
    pygame.display.update()

def agent_move(keys_pressed,agent):
        if keys_pressed[pygame.K_LEFT] and agent.x - VEL > 0:
            agent.x -= VEL
        if keys_pressed[pygame.K_RIGHT] and agent.x + VEL < GRID_SZ - AGENT_SZ:
            agent.x += VEL
        if keys_pressed[pygame.K_UP] and agent.y - VEL > 0:
            agent.y -= VEL
        if keys_pressed[pygame.K_DOWN] and agent.y - VEL < GRID_SZ - AGENT_SZ:
            agent.y += VEL
        # if keys_pressed[pygame.SPACE]
            
def handle_collision_p(agent,dots_p,keys_pressed,score):
    for dot in dots_p.values():
        if agent.colliderect(dot) and keys_pressed[pygame.K_SPACE]:
            pygame.event.post(pygame.event.Event(COLLISION_P))
            # dot.height = 0
            # dot.width = 0

def handle_collision_n(agent,dots_n,keys_pressed,score):
    for dot in dots_n.values():
        if agent.colliderect(dot) and keys_pressed[pygame.K_SPACE]:
            pygame.event.post(pygame.event.Event(COLLISION_N))
            # dot.height = 0
            # dot.width = 0

def main():
    score = 0
    
    agent = pygame.Rect(GRID_SZ/2-AGENT_SZ/2, GRID_SZ/2-AGENT_SZ/2, AGENT_SZ, AGENT_SZ)
       
    dots_p = generate_dots_p(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
    dots_n = generate_dots_n(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
    
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == COLLISION_P:
                score += P_VALUE
                agent.left = GRID_SZ/2
                agent.top = GRID_SZ/2
                dots_p = generate_dots_p(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
                dots_n = generate_dots_n(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
           
            if event.type == COLLISION_N:
                score += N_VALUE
                agent.left = GRID_SZ/2
                agent.top = GRID_SZ/2
                dots_p = generate_dots_p(P_DOT_NO, GRID_RAD, DOT_SZ, P_VALUE)
                dots_n = generate_dots_n(N_DOT_NO, GRID_RAD, DOT_SZ, N_VALUE)
                
            if event.type == pygame.QUIT:
                run = False

        keys_pressed = pygame.key.get_pressed()
        agent_move(keys_pressed, agent)
        
        handle_collision_p(agent, dots_p, keys_pressed, score)
        handle_collision_n(agent, dots_n, keys_pressed, score)
     
        draw_window(agent, dots_p, dots_n, score)

    pygame.quit()

if __name__ == "__main__":
    main()
    
# prevent agent from crossing edge
# add penalty for interaction
