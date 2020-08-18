import pygame
from random import randint

branco = (255, 255, 255)
preto = (0, 0, 0)
vermelho = (255, 0, 0)
verde = (0, 255, 0)
azul = (0, 0, 255)
amarelo = (255, 255, 0)

pygame.init()

altura = 600
largura = 640
tamanho = 20

relogio = pygame.time.Clock()
fundo = pygame.display.set_mode((largura, altura))
pygame.display.set_caption('Jogo da cobrinha')
font = pygame.font.SysFont(None, 30)
font2 = pygame.font.SysFont(None, 70)

def texto(mensagem, cor, X, Y):
    texto1 = font.render(mensagem, True, cor)
    fundo.blit(texto1, [X, Y])

def texto2(mensagem, cor, X, Y):
    texto1 = font2.render(mensagem, True, cor)
    fundo.blit(texto1, [X, Y])

def apple(x, y):
    pygame.draw.rect(fundo, vermelho, [x, y, tamanho, tamanho])

def score(pontos):
    texto('Pontuação: {}'.format(pontos), azul, 250, 550)
    texto('_____________________________________________________________________________________________', azul, 0, 460)

def snake(snakeXY, cor):
    for XY in snakeXY:
        pygame.draw.rect(fundo, cor, [XY[0], XY[1], tamanho, tamanho])

def BoostApple(x, y):
    pygame.draw.rect(fundo, amarelo, [x, y, tamanho, tamanho])

def DeathApple(x, y):
    pygame.draw.rect(fundo, verde, [x, y, tamanho, tamanho])

def jogo():
    #pygame.mixer.init()
    #pygame.mixer.music.load('sonic.mp3')

    instruções = True
    rodando = False
    GameOver = False

    snake_x = randint(0, (largura - tamanho) / tamanho) * tamanho
    snake_y = randint(0, (480 - tamanho) / tamanho) * tamanho
    apple_x = randint(0, (largura - tamanho) / tamanho) * tamanho
    apple_y = randint(0, (480 - tamanho) / tamanho) * tamanho
    BoostApple_x = randint(0, (largura - tamanho) / tamanho) * tamanho
    BoostApple_y = randint(0, (480 - tamanho) / tamanho) * tamanho
    DeathApple_x = randint(0, (largura - tamanho) / tamanho) * tamanho
    DeathApple_y = randint(0, (480 - tamanho) / tamanho) * tamanho

    velocidade_y = 0
    velocidade_x = 0

    snakeXY = []
    snakeLen = 1
    pontos = 1
    velocidade = 10
    while instruções:
        fundo.fill(branco)
        texto('INSTRUÇÕES', preto, 250, 50)
        texto('MAÇÃ VERMELHA: Torna a cobra 1 bloco maior', vermelho, 50, 100)
        texto('MAÇÃ VERDE: Mata a cobra', verde, 50, 130)
        texto('MAÇÃ AMARELA: Torna a cobra 3 blocos maior', amarelo, 50, 160)
        pygame.draw.rect(fundo, preto, [240, 200, 140, 30])
        texto('COMEÇAR(C)', branco, 245, 205)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    #pygame.mixer.music.play(100)
                    rodando = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                x = pygame.mouse.get_pos()[0]
                y = pygame.mouse.get_pos()[1]
                if x > 240 and y > 200 and x < 380 and y < 230:
                    #pygame.mixer.music.play(100)
                    rodando = True

        while rodando:
            while GameOver:
                #pygame.mixer.music.stop()
                fundo.fill(branco)
                texto2('GAME OVER', vermelho, 170, 120)
                texto('Parabéns, você fez {} pontos!'.format(pontos), preto, 170, 200)
                pygame.draw.rect(fundo, preto, [130, 260, 160, 30])
                texto('CONTINUAR(C)', branco, 135, 265)
                pygame.draw.rect(fundo, preto, [400, 260, 90, 30])
                texto('SAIR(S)', branco, 410, 265)
                
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_c:
                            #pygame.mixer.music.play()
                            jogo()
                        if event.key == pygame.K_s:
                            exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x = pygame.mouse.get_pos()[0]
                        y = pygame.mouse.get_pos()[1]
                        if x > 130 and y > 260 and x < 290 and y < 290:
                            #pygame.mixer.music.play(100)
                            jogo()
                        if x > 400 and y > 260 and x < 490 and y < 290:
                            exit()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and velocidade_x != tamanho:
                        velocidade_x = -tamanho
                        velocidade_y = 0
                    if event.key == pygame.K_RIGHT and velocidade_x != -tamanho:
                        velocidade_x = tamanho
                        velocidade_y = 0
                    if event.key == pygame.K_UP and velocidade_y != tamanho:
                        velocidade_x = 0
                        velocidade_y = -tamanho
                    if event.key == pygame.K_DOWN and velocidade_y != -tamanho:
                        velocidade_x = 0
                        velocidade_y = tamanho

            fundo.fill(branco)

            snakeHead = []
            snakeHead.append(snake_x)
            snakeHead.append(snake_y)
            snakeXY.append(snakeHead)
            if len(snakeXY) > snakeLen:
                del snakeXY[0]

            if any(Bloco == snakeHead for Bloco in snakeXY[:-1]):
                GameOver = True

            score(pontos)
            snake(snakeXY, preto)
            if snake_x == apple_x and snake_y == apple_y:
                apple_x = randint(0, (largura - tamanho) / tamanho) * tamanho
                apple_y = randint(0, (480 - tamanho) / tamanho) * tamanho
                snakeLen += 1
                pontos += 1

            if snake_x == BoostApple_x and snake_y == BoostApple_y:
                BoostApple_x = randint(0, (largura - tamanho) / tamanho) * tamanho
                BoostApple_y = randint(0, (480 - tamanho) / tamanho) * tamanho
                snakeLen += 3
                pontos += 3

            if snake_x == DeathApple_x and snake_y == DeathApple_y:
                DeathApple_x = randint(0, (largura - tamanho) / tamanho) * tamanho
                DeathApple_y = randint(0, (480 - tamanho) / tamanho) * tamanho
                GameOver = True

            if snakeLen % 8 == 0:
                BoostApple(BoostApple_x, BoostApple_y)

            if snakeLen % 7 == 0:
                DeathApple(DeathApple_x, DeathApple_y)

            apple(apple_x, apple_y)

            snake_x += velocidade_x
            snake_y += velocidade_y
            pygame.display.update()
            relogio.tick(velocidade)

            if snakeLen >= 10 and snakeLen < 20:
                velocidade = 14
            if snakeLen >= 20 and snakeLen < 30:
                velocidade = 14
            if snakeLen >= 30 and snakeLen < 40:
                velocidade = 16
            if snakeLen >= 40 and snakeLen < 50:
                velocidade = 18
            if snakeLen >= 50 and snakeLen < 60:
                velocidade = 20
            if snake_x > 640 or snake_x < 0:
                GameOver = True
            if snake_y > 480 or snake_y < 0:
                GameOver = True

jogo()
