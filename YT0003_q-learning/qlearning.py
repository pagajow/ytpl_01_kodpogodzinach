import numpy as np
import json


class Qlearn:
    
    def __init__(self, rozmiar_siatki:int = 5, cel:tuple=(3, 3), przeszkody:list = [(1, 2), (2, 2), (3, 2), (3,1)]) -> None:
        # ustawienie środowiska:
        self.rozmiar_siatki = rozmiar_siatki
        self.cel = cel
        self.przeszkody = przeszkody
        
        self.nagrody = np.zeros((self.rozmiar_siatki, self.rozmiar_siatki))
        self.nagrody[self.cel] = 10
        for przeszkoda in przeszkody:
            self.nagrody[przeszkoda] = -10

        # tablica Q:
        self.Q = np.zeros((self.rozmiar_siatki, self.rozmiar_siatki, 4))  # Dla każdego stanu i akcji

        # parametry uczenia:
        self.liczba_epizodow = 300
        self.max_kroki_na_epizod = 50
        self.alpha = 0.1
        self.gamma = 0.99

    def wybierz_akcje(self, epizod, liczba_epizodow, stan):
        epsilon = 1 - epizod/liczba_epizodow
        
        if np.random.uniform(0, 1) < epsilon:
            akcja = np.random.choice([0, 1, 2, 3])
        else:
            akcja = np.argmax(self.Q[stan])
        return akcja


    def wykonaj_akcje(self, stan, akcja):
        ruchy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nowy_stan = (max(0, min(self.rozmiar_siatki - 1, stan[0] + ruchy[akcja][0])), max(0, min(self.rozmiar_siatki - 1, stan[1] + ruchy[akcja][1])))
        return nowy_stan


    def trenuj(self):
        print("trenuję...")
        for epizod in range(self.liczba_epizodow):
            stan = (0, 0)
            for krok in range(self.max_kroki_na_epizod):
                akcja = self.wybierz_akcje(epizod, self.liczba_epizodow, stan)
                nowy_stan = self.wykonaj_akcje(stan, akcja)
                nagroda = self.nagrody[nowy_stan]
                
                self.Q[stan][akcja] += self.alpha * (nagroda + self.gamma * np.max(self.Q[nowy_stan]) - self.Q[stan][akcja])
                
                stan = nowy_stan
                if stan == self.cel or nagroda == -10:
                    break
        print("zakończyłem trening.")
        
    def testuj(self):
        dane = {
            "liczba_epizodow": self.liczba_epizodow, 
            "cel": self.cel,
            "przeszkody": self.przeszkody,
            "stany": []}
        print("\ntestuję agenta:")
        stan = (0, 0)
        for krok in range(self.max_kroki_na_epizod):
            dane["stany"].append(stan)
            akcja = self.wybierz_akcje(self.liczba_epizodow, self.liczba_epizodow, stan)
            nowy_stan = self.wykonaj_akcje(stan, akcja)
            nagroda = self.nagrody[nowy_stan]
            #Q[stan][akcja] += alpha * (nagroda + gamma * np.max(Q[nowy_stan]) - Q[stan][akcja])
            print(f"Krok {krok}: {stan} -({akcja})-> {nowy_stan}, Q: {self.Q[stan][akcja]} / {self.Q[stan]}")
            stan = nowy_stan
            if stan == self.cel or nagroda == -10:
                print(f"KONIEC: {'cel osiągniety' if stan == self.cel else 'przeszkoda..'}")
                break
        dane["stany"].append(stan)
        return dane

def zapisz_do_pliku(dane, plik="date.txt"):
    with open(plik, 'a') as f:
        f.write(json.dumps(dane)+"\n")

if __name__ == "__main__":
    q = Qlearn()
    q.trenuj()
    zapisz_do_pliku(q.testuj())

