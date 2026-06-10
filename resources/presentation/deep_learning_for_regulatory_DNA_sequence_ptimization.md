---
marp: true
theme: default
paginate: true
_paginate: false
backgroundColor: #ffffff
color: #333333
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 40px;
  }
  h1 {
    color: #003366;
    font-size: 1.8em;
  }
  h2 {
    color: #555555;
    font-size: 1.2em;
  }
  footer {
    font-size: 0.5em;
    color: #777777;
  }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
---

# OmniPhysiBoSS

**Joanna Huba** **Max Stróżyk**
Faculty of Mathematics, Informatics and Mechanics
University of Warsaw

*June 10, 2026*

---

# 1. przykładowy slajd 

- **Current Paradigm:** State-of-the-art pipelines use pre-defined, specialized biological models.
- **The Core Problem:** Simulating single, hardcoded disease pathways ignores the global context of the tissue, hiding potential systemic side effects.
- **My Goal:** Develop an open, adaptable framework (`OmniPhysiBoSS`) capable of building dynamically structured tissue models from scratch using broad network databases and user-provided multi-omic data.

---







Max Schema:
Część wstęp 
1. Wstęp jaki był problem
2. Szybkie omówienie z czego startowaliśmy (ile modeli, +- jakie, jakie dobrać intepretery oraz optimizery)
Cześć metodologia - architektura
3. Potem workflow / architekture opowiedzieć
   1. najpirew jak przecodizmy z informacjami, w jaki sposób to zaagregowalismy żeby całośc dziąła
   2. tutaj wspomnienie o interpreterach oraz opitmizeatch jak to nazywa do czego służa
      1. skomentować co zrobiliśmy (to ważne):
            1. że nie mutaliwiśmy wstęcznei - wyciszaliśmy miesjca po mutacjach (zerowaliśmy score dla optimizerów)_ (dodał rano) 
            2. (o tych mutacjach co zaimplentowałaś)
            3. i co zbieraliśmy (ważne żę zbieraliśmy WAGI I CAŁĘ BEAMY SEKEWNDCJE - też dodałem - co nam pozwoliło badać rozkłądy)
            4. ... (i że całość jest zoptymaliozwa - wszystko jest wektorowo na batchach liczone) - tego nei trzeba bo jesteśmy spóxnien xd więc ta szybkosć to paradoks w tym przypadku 
            5. A ten baseliene do stiocastycznego to dawałem losową sekwencje która jest +- podobna do wyjściowej 5-10 % odhcyelnai (nie pamieam jest w skrypcie funkcja która to robi) z dokładnym opisem 
            6.  
      2. Jak coś jescze chcesz dodać do swojej części ??? 
      3. I na koniec wszystko zapisujemy w (troche skasowanym) json - później c powiem co zrobiliśmy źle i co było złe podczas wczytywania ale to dopiero później zrozumiałem
4. Metodologia - analiza modeli 
   1. Tutaj ja wypłeniam będzie o 
      1. jak koszt wybraliźmy 
      2. które intepretry optmizery najlepije ogarniaja
      3. potem wybór sekwencji jak już znamy narzędzia
   2. i potem o ostaecznej decyzji jaka padła 
5. podsumowanie 

