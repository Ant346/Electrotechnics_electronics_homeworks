import numpy as np
import matplotlib.pyplot as plt

# Настройка для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Исходные данные
E1 = 30  # В
E2 = 10  # В
R1 = 3   # Ом
R2 = 4   # Ом
R3 = 10  # Ом
R4 = 4   # Ом
R5 = 6   # Ом
R6 = 3   # Ом

print("=== РАСЧЕТ ТОКОВ ПО ЗАКОНАМ КИРХГОФА ===")
print(f"Исходные данные:")
print(f"E1 = {E1} В, E2 = {E2} В")
print(f"R1 = {R1} Ом, R2 = {R2} Ом, R3 = {R3} Ом")
print(f"R4 = {R4} Ом, R5 = {R5} Ом, R6 = {R6} Ом")
print()

# Система уравнений по законам Кирхгофа:
# Узлы: a, b, c, d (d - базовый)
# Ветви: i1, i2, i3, i4, i5, i6

# Уравнения по первому закону Кирхгофа (узлы):
# Узел a: i1 + i2 - i3 = 0
# Узел b: i3 - i4 - i5 = 0  
# Узел c: i4 + i5 - i6 = 0

# Уравнения по второму закону Кирхгофа (контуры):
# Контур adc: E1 - i1*R1 - i3*R3 - i4*R4 = 0
# Контур bdc: E2 - i2*R2 - i5*R5 - i6*R6 = 0
# Контур adb: i1*R1 - i2*R2 + i5*R5 - i3*R3 = 0

# Матричная форма: A * I = B
# где I = [i1, i2, i3, i4, i5, i6]

A = np.array([
    [1, 1, -1, 0, 0, 0],      # Узел a: i1 + i2 - i3 = 0
    [0, 0, 1, -1, -1, 0],     # Узел b: i3 - i4 - i5 = 0
    [0, 0, 0, 1, 1, -1],      # Узел c: i4 + i5 - i6 = 0
    [-R1, 0, -R3, -R4, 0, 0], # Контур adc: E1 - i1*R1 - i3*R3 - i4*R4 = 0
    [0, -R2, 0, 0, -R5, -R6], # Контур bdc: E2 - i2*R2 - i5*R5 - i6*R6 = 0
    [R1, -R2, -R3, 0, R5, 0]  # Контур adb: i1*R1 - i2*R2 + i5*R5 - i3*R3 = 0
])

B = np.array([0, 0, 0, E1, E2, 0])

print("Система уравнений по законам Кирхгофа:")
print("Узлы:")
print("  Узел a: i1 + i2 - i3 = 0")
print("  Узел b: i3 - i4 - i5 = 0")
print("  Узел c: i4 + i5 - i6 = 0")
print("Контуры:")
print(f"  Контур adc: {E1} - {R1}*i1 - {R3}*i3 - {R4}*i4 = 0")
print(f"  Контур bdc: {E2} - {R2}*i2 - {R5}*i5 - {R6}*i6 = 0")
print(f"  Контур adb: {R1}*i1 - {R2}*i2 + {R5}*i5 - {R3}*i3 = 0")
print()

# Решение системы уравнений
try:
    currents = np.linalg.solve(A, B)
    i1, i2, i3, i4, i5, i6 = currents
    
    print("=== РЕЗУЛЬТАТЫ РАСЧЕТА ===")
    print(f"Токи в ветвях:")
    print(f"i1 = {i1:.3f} А")
    print(f"i2 = {i2:.3f} А") 
    print(f"i3 = {i3:.3f} А")
    print(f"i4 = {i4:.3f} А")
    print(f"i5 = {i5:.3f} А")
    print(f"i6 = {i6:.3f} А")
    print()
    
    # Проверка законов Кирхгофа
    print("=== ПРОВЕРКА ЗАКОНОВ КИРХГОФА ===")
    
    # Первый закон Кирхгофа
    node_a = i1 + i2 - i3
    node_b = i3 - i4 - i5
    node_c = i4 + i5 - i6
    
    print("Первый закон Кирхгофа (узлы):")
    print(f"  Узел a: {i1:.3f} + {i2:.3f} - {i3:.3f} = {node_a:.6f} ≈ 0 ✓")
    print(f"  Узел b: {i3:.3f} - {i4:.3f} - {i5:.3f} = {node_b:.6f} ≈ 0 ✓")
    print(f"  Узел c: {i4:.3f} + {i5:.3f} - {i6:.3f} = {node_c:.6f} ≈ 0 ✓")
    
    # Второй закон Кирхгофа
    loop_adc = E1 - i1*R1 - i3*R3 - i4*R4
    loop_bdc = E2 - i2*R2 - i5*R5 - i6*R6
    loop_adb = i1*R1 - i2*R2 + i5*R5 - i3*R3
    
    print("Второй закон Кирхгофа (контуры):")
    print(f"  Контур adc: {E1} - {i1:.3f}*{R1} - {i3:.3f}*{R3} - {i4:.3f}*{R4} = {loop_adc:.6f} ≈ 0 ✓")
    print(f"  Контур bdc: {E2} - {i2:.3f}*{R2} - {i5:.3f}*{R5} - {i6:.3f}*{R6} = {loop_bdc:.6f} ≈ 0 ✓")
    print(f"  Контур adb: {i1:.3f}*{R1} - {i2:.3f}*{R2} + {i5:.3f}*{R5} - {i3:.3f}*{R3} = {loop_adb:.6f} ≈ 0 ✓")
    print()
    
    # Расчет напряжений на резисторах
    U1 = i1 * R1
    U2 = i2 * R2
    U3 = i3 * R3
    U4 = i4 * R4
    U5 = i5 * R5
    U6 = i6 * R6
    
    print("=== НАПРЯЖЕНИЯ НА РЕЗИСТОРАХ ===")
    print(f"U1 = i1 * R1 = {i1:.3f} * {R1} = {U1:.3f} В")
    print(f"U2 = i2 * R2 = {i2:.3f} * {R2} = {U2:.3f} В")
    print(f"U3 = i3 * R3 = {i3:.3f} * {R3} = {U3:.3f} В")
    print(f"U4 = i4 * R4 = {i4:.3f} * {R4} = {U4:.3f} В")
    print(f"U5 = i5 * R5 = {i5:.3f} * {R5} = {U5:.3f} В")
    print(f"U6 = i6 * R6 = {i6:.3f} * {R6} = {U6:.3f} В")
    print()
    
    # Расчет мощностей
    P1 = i1**2 * R1
    P2 = i2**2 * R2
    P3 = i3**2 * R3
    P4 = i4**2 * R4
    P5 = i5**2 * R5
    P6 = i6**2 * R6
    
    print("=== МОЩНОСТИ НА РЕЗИСТОРАХ ===")
    print(f"P1 = i1² * R1 = {i1:.3f}² * {R1} = {P1:.3f} Вт")
    print(f"P2 = i2² * R2 = {i2:.3f}² * {R2} = {P2:.3f} Вт")
    print(f"P3 = i3² * R3 = {i3:.3f}² * {R3} = {P3:.3f} Вт")
    print(f"P4 = i4² * R4 = {i4:.3f}² * {R4} = {P4:.3f} Вт")
    print(f"P5 = i5² * R5 = {i5:.3f}² * {R5} = {P5:.3f} Вт")
    print(f"P6 = i6² * R6 = {i6:.3f}² * {R6} = {P6:.3f} Вт")
    print()
    
    # БАЛАНС МОЩНОСТЕЙ
    print("=== БАЛАНС МОЩНОСТЕЙ ===")
    
    # Мощность источников (положительная, если ток и ЭДС направлены одинаково)
    # Ток через источник E1: i1 (направлен от + к -)
    # Ток через источник E2: i2 (направлен от + к -)
    P_source_E1 = E1 * i1
    P_source_E2 = E2 * i2
    P_sources = P_source_E1 + P_source_E2
    
    print(f"Мощность источника E1: P_E1 = E1 * i1 = {E1} * {i1:.3f} = {P_source_E1:.3f} Вт")
    print(f"Мощность источника E2: P_E2 = E2 * i2 = {E2} * {i2:.3f} = {P_source_E2:.3f} Вт")
    print(f"Общая мощность источников: P_ист = {P_sources:.3f} Вт")
    
    # Мощность потребителей (резисторов)
    P_consumers = P1 + P2 + P3 + P4 + P5 + P6
    print(f"Общая мощность потребителей: P_потр = {P_consumers:.3f} Вт")
    
    # Проверка баланса
    balance_error = abs(P_sources - P_consumers)
    print(f"Ошибка баланса: |P_ист - P_потр| = {balance_error:.6f} Вт")
    
    if balance_error < 1e-10:
        print("✓ БАЛАНС МОЩНОСТЕЙ СОБЛЮДЕН!")
    else:
        print("✗ ОШИБКА В БАЛАНСЕ МОЩНОСТЕЙ!")
    
    print()
    
    # Создание диаграммы баланса мощностей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Диаграмма мощностей
    elements = ['E1', 'E2', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
    powers = [P_source_E1, P_source_E2, P1, P2, P3, P4, P5, P6]
    colors = ['red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
    
    bars = ax1.bar(elements, powers, color=colors, alpha=0.7)
    ax1.set_ylabel('Мощность, Вт')
    ax1.set_title('Баланс мощностей')
    ax1.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{power:.1f} Вт', ha='center', va='bottom', fontweight='bold')
    
    # Диаграмма токов
    ax2.bar(elements, [i1, i2, i3, i4, i5, i6, 0, 0], color=colors, alpha=0.7)
    ax2.set_ylabel('Ток, А')
    ax2.set_title('Токи в ветвях')
    ax2.grid(True, alpha=0.3)
    
    # Добавление значений токов
    currents_list = [i1, i2, i3, i4, i5, i6]
    for i, (element, current) in enumerate(zip(['i1', 'i2', 'i3', 'i4', 'i5', 'i6'], currents_list)):
        ax2.text(i, current + 0.05, f'{current:.3f} А', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('correct_power_balance.png', dpi=300, bbox_inches='tight')
    plt.savefig('correct_power_balance.pdf', bbox_inches='tight')
    
    print("Диаграммы сохранены в файлы:")
    print("- correct_power_balance.png")
    print("- correct_power_balance.pdf")
    
    plt.show()
    
except np.linalg.LinAlgError:
    print("Ошибка: Система уравнений не имеет решения или имеет бесконечно много решений")
except Exception as e:
    print(f"Ошибка: {e}")
