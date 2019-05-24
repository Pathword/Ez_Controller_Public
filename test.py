import control as c
import numpy as np


asdf = c.tf([1,0,0],[1,2,3,0])
t = np.linspace(0,5,1000)

print(c.step_info(asdf,t))
