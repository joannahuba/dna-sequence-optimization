# Deep Learning for Regulatory DNA Sequence Optimization

For this final group project, you will transition from evaluating sequences to engineeringthem, using your deep learning models as a "simulation environment" to either repair or engineer synthetic
DNA sequences.

***

## Introduction

In this repostitory we created  computational engineering pipeline designed to restore disrupted enhancer sequences and optimize synthetic promoter activity. Using deep learning models as an *in silico* simulation environment, we navigate the complex cis-regulatory grammar to maximize or recover the `rna_dna_ratio`. 

To achieve this, we have developed a modular, structured library architecture that enables independent manipulation of individual components, facilitating high-throughput testing across diverse models.

### Key Features
#TODO - co zaimplementowalismy 

***

## Usage

### Installation

***

### Tutorial (? - how to run it)_ 

***

## Authors

#TODO - wypisać nas i co kto zrobił 




*** 

# Notatki| Raport 

Założenia 

My reprezentujemy nasze sekwencje jako 4 klasy (one hot encoding) więc nasza przestrzeń końcowa ma postać $\mathbb{R}^{L\times 4}$. Optymalizacje przeprowadzamy poprzez znajdywanie znanych motywów  

## Cel
Będziemy chcieli **tworzyć sekwencje** - zatem musimy zrobić model generatywny która *naprawia* (modyfikuje istniejące) lub *tworzy* (od zera generujemy sekwencje) 
### Zadanie 1
**Cel**
Odzsakać i **poprawić** aktynwośc regulatorową sekwencji 
**Dane początkowe:**
- mamy popsute sekwencje
- oryginalna aktywność i ile mutacji zostało wprowadzonych

### Zadanie 2
**Cel**
Zrobić model który **iteracyjnie** poprawia tą sekwencje poprzez zwrotną informacje z modelu. 
**Dane poczatkowe**
- mało aktywna sekwencja

## Modele 

### Joanna
...

### Optimizer z integrated gradients.
#### Idea
Ten *optimizer* ma zapobiegać 

##### Definicja matematyczna
Dla pozycji $i$ oraz nukleotydu $j$ wartość IG wynosi:

$$\text{IG}_{i,j}(X) = (X_{i,j} - X'_{i,j}) \times \int_{0}^{1} \frac{\partial f(X' + \alpha(X - X'))}{\partial X_{i,j}} d\alpha$$

> UWAGA:
> Tutaj 

W praktyce całkę przybliża się sumą Riemannowską dla $m$ dyskretnych kroków (np. $m = 50$):

$$\text{IG}_{i,j}^{\text{approx}}(X) = (X_{i,j} - X'_{i,j}) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial f\left(X' + \frac{k}{m}(X - X')\right)}{\partial X_{i,j}}$$

***
PONIŻEJ SUCHE PROPOZYCJE BYŁY

### Optymalizacja po jednej sekwencji

Optymalizujemy po jednej sekwencji ( tam stosujemy odpwoeidnio model) następnie wybieramy do jakiego scoer'u ma dojść / maksymalną modyfikacje isntiejącej sekwencji

#### Wersja z mutacjami (losowo zmieniamy sekwencje (przed / po optymalizacji ??) w celu ucieknięca z minimum lokalnego)

### Optmyalizacja po wielu sekwencjach naraz

Wybieramy ile ma być jednocześnie zmienianych mutacji i otrzymujemy co sie zmienia 

### Była wersja z wykrywniem motifów,
Wiemy że to jest człowiek, więc możeym zrobić tak, że wyszukujemy znanych wzórców i próbujemy je dopasować. albo że wogóle, tylko motify znajdujemy w sekwencji i tylko te motify pozwalami modeli optymalizować zmieniać ?? względem jakiś wzorców z dostępnej bazy danych motifów (człowiek??) - dodatkowe jak starczy czasu bo to jest ciekawe 



## Techniczne przed analizą 
### modele do sprawdzenia 
- jak już na dwóch modelach będzie przygotowaqna całą analiza to wtedy spróbujemy jeszcze wstawić kilka moich innych modeli, które miał +- sensowne score'y - trzeba zrobić tabele która je porównuje na jakiś wspólnym zbiorze testowym

### Skrypty do zrobienia
- funkcja która wylicza score'y po wskazanych modelahc i zwraca z tego df'a model walidacyjny | score sekwencji
  - będzie on uruchamiał 
-  (jeśli nie ma)



***

## Główna analiza 

### Część I - konstrukcja eksperymentu 

#### Dobór modeli walidacyjnych 
Tutaj decydujemy które modele będą brane podczas analizy zachowują się podobnie, czyli tak: 
- robimy trajektorie zmian, sprawdzamy (**Rozkłady / średnie** po wszystkich modelach) i sprawdzamy czy wogóle model sensownie optymalizuje czy jest tak rozjechany względem pozostałych (przy kilku modelach jest bardzo nei prwadopodobne żeby otrzymać nmieskei score' - sprawdzamy na które modele które służa jako ground truth uważać) 
  - czyli labelujemy po tym jaki model był walidacyjny 
  - na x są trajektorie zmiany (każdy model osobno bo zalężnie od modleu mamy inne sekwencje, ale własnie o to chodzi badamy jak modele się zachowują względem iteracji)
  - na $y$ rozpisujemy średnie takich modeli (rozkłady będzie błędów trzeba by chyba zrobić na osobnym plotcie żeby to widoczne) 
  
Wszystkie wyniki będziemy rozpatrywać na wyznaczonej grupie modeli, które sprawdziliśmy jak się zachowują. Te które znacząco się rozjeżdżają jeżeli chodzi o średnie / rozkłady score'ów zostają odrzucone 


> Chyba tylko tyle czy coś jeszcze tutaj będzie potrzebne 

### Część II - badanie modeli

### Optimizery 

#### Sprawdzić jak bardzo się różnią sekwencje (możemy miec wiele propozycji gdzie wszystkie trafiają w jedno miejsce)
chcemy sprawdzić


#### średni ostateczny score sekwencji ??
- to jest słąb

### Interpretery 

#### Problem trafienia w silny czynnik transkrypcyjny

Może się zdarzyć, że model wpadnie w motif który znacznie wpływa na ekspresje, wtedy sekwencja będzie zmieniać losowe sekwencje ponieważ wszystkie zmiany będą ~ 0 


### Część III - rzetelność sekwencji ??? 


# Co do zmiany 

interpretation -> interpreters !!! 
- według konwencji. 




# Notatki

