# Restore methods
We use statistic methods:
* navier_stokes 
* telea
* bilinear
* nearest_neighbor
* lanczos
* bicubic
* gauss

Tests provided on images from Cifar-10 Dataset (32x32 px)

____
## Result of comparison of statistical methods
### On Attacked by one px attack images
Result from analyze on 151 images.

| Method                    | Score  | Quality | Time,<br/>sec*10^3 |
|---------------------------|--------|---------|--------------------|
| Фильтр Гаусса             | 2.7152 | 0.8781  | 0.887              |
| Уравнение Навье-Стокса    | 2.7417 | 0.8756  | 0.052              |
| Метод Талеа               | 2.7682 | 0.8720  | 0.036              |
| Билинейная интерполяция   | 3.3046 | 0.8717  | 0.007              |
| Метод ближайшего соседа   | 3.4371 | 0.8520  | 0.004              |
| Фильтр Ланцоша            | 3.8146 | 0.8590  | 2.608              |
| Бикубическая интерполяция | 5.3709 | 0.7622  | 0.198              |


