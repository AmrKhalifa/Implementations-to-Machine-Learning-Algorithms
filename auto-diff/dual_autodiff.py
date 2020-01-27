import math 

class Dual:

	def __init__(self, a, b):
		self.real = a
		self.dual = b
	
	def __add__(self, b):
		real = self.real + b.real
		dual = self.dual + b.dual
		return Dual(real, dual)

	def __neg__(self):
		return Dual(-self.real, -self.dual)

	def __sub__(self, b):
		return self.__add__(-b)

	def __mul__(self, b):
		real = self.real * b.real
		dual = self.real*b.dual + self.dual*b.real 
		return Dual(real, dual) 

	def __truediv__(self, b):
		assert b.real !=0 , 'Division by Zero Error'
		real = self.real/b.real 
		dual = (self.dual*b.real - self.real*b.dual)/(b.real**2)
		return Dual(real, dual) 

	def __str__(self):
		if self.dual >= 0:
			representation = str(self.real)+'+'+str(self.dual)+ "\u03B5"
		else:
			representation = str(self.real)+str(self.dual)+ "\u03B5"
		return representation

def sin(a):
	return Dual(math.sin(a.real), math.cos(a.dual)*a.dual)
def cos(a):
	return Dual(math.cos(a.real), -math.sin(a.dual)*a.dual) 
def tan(a):
	return sin(a)/cos(a) 
def exp(a):
	return Dual(math.exp(a.real), math.exp(a.real)*a.dual)
def log(a):
	assert a.real >0, 'Math Error, log(0 or negative)'
	return Dual(math.log(a.real), a.dual/a.real)
def sqrt(a):
	assert a.real != 0, 'Division by zero error'
	sq = math.sqrt(a.real)
	return Dual(sq, a.dual/(2*sq))

x = Dual(1, 2)
y = Dual(1, 4)
print(x/y)