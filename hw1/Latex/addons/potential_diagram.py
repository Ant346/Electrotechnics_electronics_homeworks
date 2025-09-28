import matplotlib.pyplot as plt
import numpy as np

# Настройка для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Данные из расчета
# Потенциалы узлов (из метода узловых потенциалов)
phi_a = 22.5  # В
phi_b = 6.25  # В  
phi_c = 2.5   # В
phi_d = 0     # В (базовый узел)

# Токи в ветвях
i1 = 2.5   # А
i2 = 1.25  # А
i3 = 1.25  # А
i4 = 1.25  # А
i5 = 2.5   # А
i6 = 1.25  # А

# Сопротивления
R1 = 3   # Ом
R2 = 4   # Ом
R3 = 10  # Ом
R4 = 4   # Ом
R5 = 6   # Ом
R6 = 3   # Ом

# ЭДС источников
E1 = 30  # В
E2 = 10  # В

# Создание фигуры
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 1. Потенциальная диаграмма узлов
nodes = ['a', 'b', 'c', 'd']
potentials = [phi_a, phi_b, phi_c, phi_d]

ax1.bar(nodes, potentials, color=['red', 'blue', 'green', 'black'], alpha=0.7)
ax1.set_ylabel('Потенциал, В')
ax1.set_title('Потенциальная диаграмма узлов')
ax1.grid(True, alpha=0.3)

# Добавление значений на столбцы
for i, (node, pot) in enumerate(zip(nodes, potentials)):
    ax1.text(i, pot + 0.5, f'φ{node} = {pot} В', ha='center', va='bottom', fontweight='bold')

# 2. Потенциальная диаграмма по контуру
# Выберем контур adbc (через узлы a-d-b-c)
contour_nodes = ['a', 'd', 'b', 'c']
contour_potentials = [phi_a, phi_d, phi_b, phi_c]

# Построение потенциальной диаграммы по контуру
x_positions = np.arange(len(contour_nodes))
ax2.plot(x_positions, contour_potentials, 'ro-', linewidth=3, markersize=8, label='Потенциалы узлов')
ax2.set_xlabel('Узлы контура')
ax2.set_ylabel('Потенциал, В')
ax2.set_title('Потенциальная диаграмма по контуру adbc')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(contour_nodes)

# Добавление значений потенциалов
for i, (node, pot) in enumerate(zip(contour_nodes, contour_potentials)):
    ax2.annotate(f'φ{node} = {pot} В', 
                xy=(i, pot), xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Добавление информации о токах и напряжениях
info_text = f"""
Данные из расчета:
Токи: i₁={i1}А, i₂={i2}А, i₃={i3}А, i₄={i4}А, i₅={i5}А, i₆={i6}А
Напряжения на резисторах:
U₁ = i₁×R₁ = {i1}×{R1} = {i1*R1} В
U₂ = i₂×R₂ = {i2}×{R2} = {i2*R2} В  
U₃ = i₃×R₃ = {i3}×{R3} = {i3*R3} В
U₄ = i₄×R₄ = {i4}×{R4} = {i4*R4} В
U₅ = i₅×R₅ = {i5}×{R5} = {i5*R5} В
U₆ = i₆×R₆ = {i6}×{R6} = {i6*R6} В
"""

# Добавление текстовой информации
fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Оставляем место для текста

# Сохранение в файл
plt.savefig('potential_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('potential_diagram.pdf', bbox_inches='tight')

print("Потенциальная диаграмма сохранена в файлы:")
print("- potential_diagram.png")
print("- potential_diagram.pdf")

plt.show()
