# Circuit cutting playground
## Содержание
Данный репозиторий создан для хранения кода, реализующего протоколы разреза квантовых цепочек.

## Текущее состояние 
1. На данный момент реализовано _КАК_ разложение двухкубитных унитарных матриц в исходной форме 
$$U = a_0 \otimes a_1 \cdot exp(i \phi_x XX + i \phi_y YY + i \phi_z ZZ) \cdot b_o \otimes b_1$$
И в форме так называемого _KAK-like_ разложения 
$$U = a_0 \otimes a_1 \cdot (\sum_{i=0}^3 u_i \sigma_i \otimes \sigma_i) \cdot b_0 \otimes b_1 := a_0 \otimes a_1 \cdot W \cdot b_0 \otimes b_1$$
 
2. Реализованы функции построения цепочек, заменяющих данный гейт, по его _KAK-like_ разложению

3. В ноутбуке cutting.ipynb продемонстрирована работа кода на примере простой двухкубитной цепочки с одним двухкубитным гейтом и описан дальнейший план работ по теме.

## Источники 
1. Robert R. Tucci, An Introduction to Cartan's KAK Decomposition for QC Programmers, [arXiv:quant-ph/0507171](https://arxiv.org/abs/quant-ph/0507171)
2. Lukas Schmitt, Christophe Piveteau, David Sutter, Cutting circuits with multiple two-qubit unitaries,  
   [arXiv:2312.11638 [quant-ph]](https://arxiv.org/abs/2312.11638v3)
