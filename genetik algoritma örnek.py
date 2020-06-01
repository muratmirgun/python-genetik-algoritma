#!/usr/bin/env python3
# coding: utf-8

import sys
import math

import pygame

import random

BLUE = [20, 108, 121]
BLACK = [39,  47,  48]

simulation_options = {
    'fps': 0,
    'size': (400, 400),
    'target': (1/4, 1/8),
    'mutrate': 0.03,
    'popsize': 50,
    'lifespan': 250,
    'maxforce': random.uniform(0.2, 2),
    'obstacles': [
        pygame.Rect(0, 150, 230, 10),
        pygame.Rect(170, 250, 230, 10),
        pygame.Rect(220, 50, 10, 100)
    ]
}


def map_(n, a, b, c, d):
    return ((n-a) / (b-a)) * (d - c) + c


class Vector(object):
    def __init__(self, *args):
        self.values = args if args else (0, 0)

    def __str__(self):
        return str(self.values)

    def __add__(self, other):
        return Vector(*(p + q for p, q in zip(self, other)))

    def __sub__(self, other):
        return Vector(*(p - q for p, q in zip(self, other)))

    def __mul__(self, other):
        return Vector(*(p * q for p, q in zip(self, other)))

    def __div__(self, other):
        return Vector(*(p / q for p, q in zip(self, other)))

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def mag(self):
        return math.sqrt(sum(c ** 2 for c in self))

    def copy(self):
        return Vector(*self.values)

    def limit(self, max):
        return self if self.mag() < math.sqrt(max) else self.set_mag(max)

    def set_mag(self, n):
        return self.normalize() * Vector(n, n)

    def angle(self):
        return math.atan2(self[1], self[0])

    def tuple_int(self):
        return tuple(map(int, self.values))

    def normalize(self):
        return Vector(*(c / self.mag() for c in self))

    def distance(self, other):
        return math.sqrt(sum(pow(a - b, 2) for a, b in zip(self, other)))

    def rotate(self, angle):
        return Vector(
            math.cos(self.angle() + angle) * self.mag(),
            math.sin(self.angle() + angle) * self.mag()
        )

    @staticmethod
    def from_angle(angle, length=1):
        return Vector(
            math.cos(angle) * length,
            math.sin(angle) * length
        )

    @classmethod
    def random_vec(cls):
        return cls.from_angle(random.random() * 2 * math.pi)


class DNA(object):
    def __init__(self, genes, mutrate, maxforce):
        self.genes = [gene.set_mag(maxforce) for gene in genes]
        self.mutrate = mutrate
        self.maxforce = maxforce

    def mutation(self):
        for n in range(len(self.genes)):
            if random.random() < self.mutrate:
                self.genes[n] = Vector.random_vec().set_mag(self.maxforce)

    def crossover(self, parent):
        index = random.randint(0, len(self.genes))
        return DNA(
            self.genes[:index] + parent.genes[index:],
            self.mutrate, self.maxforce
        )

    @staticmethod
    def random_dna(n, mutrate, maxforce):
        return DNA(
            [Vector.random_vec() for _ in range(n)],
            mutrate, maxforce
        )


class Dot(object):
    def __init__(self, dna, acc, vel, pos):
        self.acc, self.vel = acc, vel
        self.dna, self.pos = dna, pos

        self.fitness = 0
        self.is_alive = True
        self.complete = False

    def calc_fitness(self, target):
        self.fitness = map_(target.distance(self.pos), 0, 400, 400, 0)

        if self.complete:
            self.fitness *= 10

        if not self.is_alive:
            self.fitness /= 10

    def update(self, gene_n, target):
        if self.pos.distance(target) < 20:
            self.pos = target.copy()
            self.complete = True

        if self.is_alive and not self.complete:
            self.acc += self.dna.genes[gene_n]
            self.vel += self.acc
            self.pos += self.vel
            self.acc *= Vector(0, 0)
            self.vel = self.vel.limit(4)


class Population(object):
    def __init__(self, init_pos, target, pop_size, lifespan, mutrate, maxforce):
        self.gene_n = 0
        self.target = target
        self.max_fit = 0
        self.init_pos = init_pos

        self.generation = 1
        self.lifespan = lifespan
        self.mating_pool = []
        self.population = self.generate_pop(
            init_pos, pop_size, lifespan, mutrate, maxforce
        )

    def __iter__(self):
        return iter(self.population)

    def evaluate(self):
        self.mating_pool.clear()
        self.max_fit = 0

        for dot in self:
            dot.calc_fitness(self.target)
            self.max_fit = dot.fitness if dot.fitness > self.max_fit else self.max_fit

        for dot in self:
            dot.fitness = dot.fitness / self.max_fit
            n = max(1, dot.fitness * 100)
            self.mating_pool += [dot] * int(n)

        print('    generation:', self.generation)
        print('   max fitness:', self.max_fit)
        print('succesful dots:', len([x for x in self if x.complete == True]))

    def selection(self):
        for n in range(len(self.population)):
            parent_a = random.choice(self.mating_pool).dna
            parent_b = random.choice(self.mating_pool).dna
            while parent_a == parent_b:
                parent_b = random.choice(self.population).dna

            child_dna = parent_a.crossover(parent_b)
            child_dna.mutation()
            self.population[n] = Dot(
                child_dna, Vector(), Vector(), Vector(*self.init_pos)
            )

    def run(self):
        for dot in self.population:
            dot.update(self.gene_n, self.target)

        self.gene_n += 1
        if self.gene_n == self.lifespan:
            self.evaluate()
            self.selection()
            self.gene_n = 0
            self.generation += 1

    def generate_pop(self, pos, size, lifespan, mutrate, maxforce):
        population = [
            Dot(
                DNA.random_dna(lifespan, mutrate, maxforce),
                Vector(),
                Vector(),
                Vector(*pos)
            ) for _ in range(size)
        ]
        return population


class Simulation(object):
    pygame.init()
    pygame.display.set_caption('smart lines')

    def __init__(self, fps, size, target, mutrate, popsize, lifespan, maxforce, obstacles):
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)

        self.w, self.h = size
        self.target = Vector(*size) * Vector(*target)
        self.obstacles = obstacles
        self.population = Population(
            (self.w / 2, self.h - 50), self.target,
            popsize, lifespan, mutrate, maxforce
        )

    def is_dead(self, dot):
        x, y = dot.pos.values
        return any([
            x < 0,
            y < 0,
            x > self.w,
            y > self.h
        ] + [rect.collidepoint(x, y) for rect in self.obstacles])

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.fps ^= (60 ^ 0)

    def draw_loop(self):
        pygame.draw.circle(self.screen, [110] * 3, self.target.tuple_int(), 20)

        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, [110] * 3, obstacle)

        for dot in self.population:
            if self.is_dead(dot):
                dot.is_alive = False

            x, y, angle = (*dot.pos.values, dot.vel.angle())
            c, d = x - 20 * math.cos(angle), y - 10 * math.sin(angle)
            pygame.draw.line(self.screen, BLACK, (x, y), (c, d), 2)

        self.population.run()

    def main_loop(self):
        while True:
            self.handle_events()
            self.screen.fill(BLUE)
            self.draw_loop()

            pygame.display.flip()
            self.clock.tick(self.fps)


if __name__ == '__main__':
    simulation = Simulation(**simulation_options)
    simulation.main_loop()
