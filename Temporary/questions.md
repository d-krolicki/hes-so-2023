1. Który z formatów należy zastosować:
    - ScalarImage z TorchIO - B:C:X:Y:Z (batch-channels-height-width-dim)
    - Proponowany - B:C:Y:X:Z (kompatybilny z PyTorch?) 
    - NumPy / LabelMap - B:C:Z:Y:X
2. Czy dane takie jak spacing lub orientation mają tutaj znaczenie?
3. Czy typ wartości w tensorach ma znaczenie poza wydajnością obliczeń?
4. Czy kodować od razu wszystkie klasy w datasecie, czy tylko te których uczymy model?
5. Skąd wziąć implementację modelu? Znalazłem:
    - MONAI
    - Czyjeś implementacje 3D UNet na GitHubie
    - Może własna implementacja w oparciu o research paper?
6. Jak przekazywać do modelu więcej niż 1 label na każdy obraz?
7. Której z implemetacji datasetu używać:
    - Implementacja PyTorcha,
    - Implementacja MONAI,
    - Implementacja TorchIO.
