# Prosta sieć neuronowa od podstaw
Projekt przedstawia prostą implementację sieci neuronowej do klasyfikacji binarnej. Sieć została stworzona w celach edukacyjnych, aby lepiej zrozumieć podstawowe mechanizmy działania sieci neuronowych:

 - Propagacja w przód: Obliczanie wyników wyjściowych na podstawie danych wejściowych i aktualnych wag i wyrazów wolnych
 - Propagacja wsteczna: Aktualizacja wag na podstawie gradientów błędu, aby poprawić dokładność modelu.
 - Funkcje aktywacji: Dodawanie nieliniowości do sieci.
 - Funkcja straty: Ocena jakości predykcji modelu.

![image](https://github.com/user-attachments/assets/a7974796-7ba7-4d0f-8ba2-f1f2d1ff5093)

Wynik sieci dla problemu binarnego:

```python
Epoka 0, Strata: 0.18732243550684546
Epoka 100, Strata: 0.17920195690671448
Epoka 200, Strata: 0.16966505792662623
Epoka 300, Strata: 0.16196923201448704
Epoka 400, Strata: 0.14713948823700024
Epoka 500, Strata: 0.12183528436100294
Epoka 600, Strata: 0.09555035086063353
Epoka 700, Strata: 0.07061970489435557
Epoka 800, Strata: 0.048175388158740576
Epoka 900, Strata: 0.030615808555348827

--Przewidywania sieci:
Input: [0 0], Wynik: 0.0031
Input: [0 1], Wynik: 0.1182
Input: [1 0], Wynik: 0.1181
Input: [1 1], Wynik: 0.7812
