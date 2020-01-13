import random 
import math 

########### Categorical distribution with given probabilities 
multinomial = [0.1, 0.4, 0.2, 0.25, .05]

########### Binomial distribution with given parameters 
def binomial_dis(n = 10, p = .5):
	fn = math.factorial(n)
	for r in range(0, n+1):
		prop = (fn/(math.factorial(r)*math.factorial(n-r)))*pow(p,r)*pow(1-p, n-r) 
		yield prop 

########## The sampling function takes any PMF and samples from it 
def sample(PMF):
	r = random.random()
	CDF = 0 
	for rand_var, prop in enumerate(PMF):
		CDF += prop
		if CDF >= r:
			return rand_var
	return rand_var	
			
outcomes_1 = []
outcomes_2 = []
for i in range (1000):
	outcomes_1.append(sample(binomial_dis(10, 0.5)))
	outcomes_2.append(sample(multinomial))


## Plotting the result of each distribtion 
import matplotlib.pyplot as plt 

plt.figure()
plt.hist(outcomes_1)

plt.figure()
plt.hist(outcomes_2)
plt.show()
