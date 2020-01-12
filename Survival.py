import os, random
from random import randrange
import numpy as np


class person:
	
	def __init__(self, cooperatability):
		self.cooperatability = cooperatability
		self.alive = True
		self.hunger = 0
		self.thirst = 0
		self.strength = 0.5
		self.health = 1
		self.reproducability = (self.hunger + self.thirst)/ 2


	def __del__(self, instance):
		del self.value

	def sleep(self):
		self.hunger += 0.5

	def live(self, population, population_size):
		if self.hunger > 1:
			person(self.cooperatability)
			population.append(person)
		if self.hunger <= 0:
			self.alive = False

	def reproduce(self, ):


class monster:
	def __init__(self, strength, health, reproducability):
		self.cooperatability = cooperatability + 


class environment:
	def __init__(self, scarcity, population):
		self.food_scarcity = scarcity
		if self.food_scarcity == 'scarce'
			self.food_available = population * 0.25
		if self.food_scarcity == 'average'
			self.food_available = population * 0.5
		if self.food_scarcity == 'abundant'
			self.food_available = population * 1.0
		


	def eat(self, person1, person2):
		# Describe cooperative eating
		if person1.cooperatability == 1 & person2.cooperatability == 1:
			person1.hunger -= 0.5
			person2.hunger -= 0.5

		# Describe mixed cooperative eating
		if person1.cooperatability == 0 & person2.cooperatability == 1:
			person1.hunger -= 0.75
			person2.hunger += 0.25

		# Describe uncooperative eating
		if person1.cooperatability == 0 & person2.cooperatability == 0:
			random_int = randint(0, 10)
			if random_int > 4:
				person1.hunger -= 0.5
				person2.hunger += 0.25
			else:
				person2.hunger -= 0.5
				person1.hunger += 0.25
		self.available -= 2

	def restock(self):
		if self.scarcity == 'scarce'
			self.available = population * 0.25
		if self.scarcity == 'average'
			self.available = population * 0.5
		if self.scarcity == 'abundant'
			self.available = population * 0.75

			
class history:

	def __init__(self):
		self.population = []
		self.avg_hunger = []

	def update(self, population_size, avg_hunger):
		self.population.append(population_size)
		self.avg_hunger.append(avg_hunger)


def main():
	# Create a history
	history = history()

	# Create people
	people = [];
	for i in range(10):
		hostile_person = person(0)
		neutral_person = person(1)
		population.append(hostile_person)
		population.append(neutral_person)

	# Run through 
	print('How many days would you like to run through?')
	days = int(input())

	for d in range(0, days):
		for p in (range(population.length/2)):
			person1 = population[p]
			person2 = population[p + (population.length/2)]
			environment.eat(person1, person2)


		environment.restock()




if __init__ == '__main__':
	main()