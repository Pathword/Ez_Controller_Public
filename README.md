# Ez_Controller_Public
Public Ez_Controller all in one version, up to date as of 5/22/2019 4pm 

Fixed sympy/pyinstaller issue. Pyinstaller gave recursion error because of sympy's circular imports. Created a new algebraic expander using pymbolic and regex. 

.exe download: https://uccsoffice365-my.sharepoint.com/:f:/g/personal/dcopley_uccs_edu/EvK_F9GeiYJNgRemwnVZTjMB5svxodTJD65lLdNJUQZO0g?e=jRCwUC

---

Gif scrubber: download gif scrubber extension for chrome: https://chrome.google.com/webstore/detail/gif-scrubber/gbdacbnhlfdlllckelpdkgeklfjfgcmp?hl=en

Allow gif scrubber to view local files: top right, three dots -> more tools -> extensions -> gifscrubber (details) -> allow access to file URLs. 

Drag and drop the gif in a chrome browser, right click image and select gif scrubber. 

---

Known Issues: 

-Governing transfer function display sometimes not big enough, simply increase width in designer. 

-step responses that settle to 0 don't return proper settling times, makes sense because (Ts) band is derived from +-2% of steady state value. Could make it as a function of peak values, double check with Dr. A/L
