# Ez_Controller_Public
Public Ez_Controller all in one version, up to date as of 5/22/2019 4pm 

Fixed sympy/pyinstaller issue. Pyinstaller gave recursion error because of sympy's circular imports. Created a new algebraic expander using pymbolic and regex. 


---
OPTIONAL:

Gif scrubber: download gif scrubber extension for chrome: https://chrome.google.com/webstore/detail/gif-scrubber/gbdacbnhlfdlllckelpdkgeklfjfgcmp?hl=en

Allow gif scrubber to view local files: top right, three dots -> more tools -> extensions -> gifscrubber (details) -> allow access to file URLs. 

Drag and drop the gif in a chrome browser, right click image and select gif scrubber. 



---

Known Issues: 


-step responses that settle to 0 don't return proper settling times, makes sense because (Ts) band is derived from +-2% of steady state value. Could make it as a function of peak values, double check with Dr. A/L. Current workaround is editing timeresp.py in controls site-package. 


---

External Links/Other information:

Showcase 1: https://youtu.be/Fg_PnOdWBQE

Showcase 2: https://youtu.be/jk98r97xM5I

essentials of controls: Spring-Mass-Damper example https://www.youtube.com/watch?v=ej7CRAIGXow
