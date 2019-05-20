# Ez_Controller_Public
Public Ez_Controller all in one version, up to date as of 5/19/2019 8pm 

Issue: Cannot make Ez_Controller_dist.py into an executable, 
"RecursionError: maximum recursion depth exceeded while calling a Python object" when using pyinstaller --onefile Ez_Controller_dist.py

SO: https://stackoverflow.com/questions/56213063/recursion-error-maximum-recursion-depth-exceeded-while-calling-a-python-object?noredirect=1#comment99047165_56213063

Python ver 3.7, pyinstaller ver 3.4

---

Can run script and works fine, not related to issue but if the program is to be used heres an example: 

Example run (simple spring-mass-damper): 

Controller: No entry 

Plant: Numerator = 8, Denominator = (s^2)+(2*s)+9

Select mode (Step Response, Root Locus, or Bode Plot) and hit run.

---

To create an animation using Animation Step Response mode, replace a coefficent variable and vary it. 

Controller: No entry 

Plant: Numerator = 8, Denominator = (s**2)+(x*s)+9 

lower = 1 upper = 5 Places animation in current gifs folder. 


