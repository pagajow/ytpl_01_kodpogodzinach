import numpy as np


def daneTestowe(nr:int):
     wejscie = np.random.randint(0, 11, size=(nr, 3))
     wyjscie = np.array([2*a - 3*b + 0.5*c for a, b, c in wejscie])
     return wejscie, wyjscie
 

class Neuron:
     
    # Sigmoid
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_pochodna(x):
        return x * (1 - x)

    # ReLU
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    @staticmethod
    def relu_pochodna(x):
        return np.where(x <= 0, 0, 1)

    # Funkcja liniowa (dla porównania)
    @staticmethod
    def linear(x):
        return x
    @staticmethod
    def linear_pochodna(x):
        return np.ones_like(x)
    
    
    def __init__(self, liczbaWejsc:int, funkcjaAktywacji: str = None) -> None:
        # Inicjalizacja wag i biasów (dla uproszczenia przyjmujemy jeden neuron)
        self.W = np.random.rand(liczbaWejsc)  
        self.b = np.random.rand(1) 
        
        if funkcjaAktywacji == "sigmoid":
            self.funkcjaAktywacji = self.sigmoid
            self.pochodnaFunkcjiAktywacji = self.sigmoid_pochodna
        elif funkcjaAktywacji == "relu":
            self.funkcjaAktywacji = self.relu
            self.pochodnaFunkcjiAktywacji = self.relu_pochodna
        else:
            self.funkcjaAktywacji = self.linear
            self.pochodnaFunkcjiAktywacji = self.linear_pochodna
    
    # Prognozowanie - proces forward propagation
    def prognozuj(self, inputs):
        return self.funkcjaAktywacji(np.dot(inputs, self.W) + self.b)

    def trenuj(self, daneWejsciowe:np.ndarray, daneWynikowe:np.ndarray, epoki:int, tempoUczenia:float=0.001):
        historia = []
        for _ in range(epoki):
            historiaEpoki = []
            for wejscie, wynik in zip(daneWejsciowe, daneWynikowe):
                # Forward propagation
                predykcja = self.prognozuj(wejscie)
                
                # Obliczanie błędu
                blad = wynik - predykcja
                
                # Aktualizacja wag i biasu zgodnie z regułą delta i pochodną funkcji aktywacji
                self.W += tempoUczenia * blad * self.pochodnaFunkcjiAktywacji(predykcja) * wejscie
                self.b += tempoUczenia * blad * self.pochodnaFunkcjiAktywacji(predykcja)
                
                historiaEpoki.append([wynik, predykcja[0], blad[0]])
            historia.append(historiaEpoki)
        return historia

    
