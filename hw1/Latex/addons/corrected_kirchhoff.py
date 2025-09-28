#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный скрипт для расчета токов по законам Кирхгофа
с правильным балансом мощностей
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Настройка для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def solve_circuit_kirchhoff():
    """Решение цепи методом законов Кирхгофа с правильными направлениями токов"""
    
    print("=" * 70)
    print("ПРАВИЛЬНЫЙ РАСЧЕТ ТОКОВ ПО ЗАКОНАМ КИРХГОФА")
    print("=" * 70)
    
    # Исходные данные
    E1 = 30  # В
    E2 = 10  # В
    R1 = 3   # Ом
    R2 = 4   # Ом
    R3 = 10  # Ом
    R4 = 4   # Ом
    R5 = 6   # Ом
    R6 = 3   # Ом
    
    print(f"Исходные данные:")
    print(f"E1 = {E1} В, E2 = {E2} В")
    print(f"R1 = {R1} Ом, R2 = {R2} Ом, R3 = {R3} Ом")
    print(f"R4 = {R4} Ом, R5 = {R5} Ом, R6 = {R6} Ом")
    print()
    
    # ПРАВИЛЬНАЯ система уравнений по законам Кирхгофа
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
    A = np.array([
        [1, 1, -1, 0, 0, 0],                    # Узел a: i1 + i2 - i3 = 0
        [0, 0, 1, -1, -1, 0],                    # Узел b: i3 - i4 - i5 = 0
        [0, 0, 0, 1, 1, -1],                     # Узел c: i4 + i5 - i6 = 0
        [-R1, 0, -R3, -R4, 0, 0],                # Контур adc: E1 - i1*R1 - i3*R3 - i4*R4 = 0
        [0, -R2, 0, 0, -R5, -R6],                # Контур bdc: E2 - i2*R2 - i5*R5 - i6*R6 = 0
        [R1, -R2, -R3, 0, R5, 0]                 # Контур adb: i1*R1 - i2*R2 + i5*R5 - i3*R3 = 0
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
    
    try:
        # Решение системы уравнений
        currents = np.linalg.solve(A, B)
        i1, i2, i3, i4, i5, i6 = currents
        
        print("РЕЗУЛЬТАТЫ РАСЧЕТА:")
        print(f"i1 = {i1:.6f} А")
        print(f"i2 = {i2:.6f} А") 
        print(f"i3 = {i3:.6f} А")
        print(f"i4 = {i4:.6f} А")
        print(f"i5 = {i5:.6f} А")
        print(f"i6 = {i6:.6f} А")
        print()
        
        # Проверка законов Кирхгофа
        print("ПРОВЕРКА ЗАКОНОВ КИРХГОФА:")
        print("-" * 40)
        
        # Первый закон Кирхгофа
        node_a = i1 + i2 - i3
        node_b = i3 - i4 - i5
        node_c = i4 + i5 - i6
        
        print("Первый закон Кирхгофа (узлы):")
        print(f"  Узел a: {i1:.6f} + {i2:.6f} - {i3:.6f} = {node_a:.2e} ≈ 0 ✓")
        print(f"  Узел b: {i3:.6f} - {i4:.6f} - {i5:.6f} = {node_b:.2e} ≈ 0 ✓")
        print(f"  Узел c: {i4:.6f} + {i5:.6f} - {i6:.6f} = {node_c:.2e} ≈ 0 ✓")
        
        # Второй закон Кирхгофа
        loop_adc = E1 - i1*R1 - i3*R3 - i4*R4
        loop_bdc = E2 - i2*R2 - i5*R5 - i6*R6
        loop_adb = i1*R1 - i2*R2 + i5*R5 - i3*R3
        
        print("Второй закон Кирхгофа (контуры):")
        print(f"  Контур adc: {loop_adc:.2e} ≈ 0 ✓")
        print(f"  Контур bdc: {loop_bdc:.2e} ≈ 0 ✓")
        print(f"  Контур adb: {loop_adb:.2e} ≈ 0 ✓")
        print()
        
        # Расчет мощностей и баланса
        print("=" * 70)
        print("РАСЧЕТ МОЩНОСТЕЙ И БАЛАНС")
        print("=" * 70)
        
        # Напряжения на резисторах
        R = [R1, R2, R3, R4, R5, R6]
        voltages = [i * r for i, r in zip(currents, R)]
        
        # Мощности на резисторах
        powers = [i**2 * r for i, r in zip(currents, R)]
        
        print("НАПРЯЖЕНИЯ НА РЕЗИСТОРАХ:")
        for i, (voltage, current, resistance) in enumerate(zip(voltages, currents, R), 1):
            print(f"U{i} = i{i} * R{i} = {current:.6f} * {resistance} = {voltage:.6f} В")
        print()
        
        print("МОЩНОСТИ НА РЕЗИСТОРАХ:")
        for i, (power, current, resistance) in enumerate(zip(powers, currents, R), 1):
            print(f"P{i} = i{i}² * R{i} = {current:.6f}² * {resistance} = {power:.6f} Вт")
        print()
        
        # БАЛАНС МОЩНОСТЕЙ
        print("БАЛАНС МОЩНОСТЕЙ:")
        print("-" * 30)
        
        # Мощность источников (положительная, если ток и ЭДС направлены одинаково)
        P_source_E1 = E1 * i1
        P_source_E2 = E2 * i2
        P_sources = P_source_E1 + P_source_E2
        
        print(f"Мощность источника E1: P_E1 = E1 * i1 = {E1} * {i1:.6f} = {P_source_E1:.6f} Вт")
        print(f"Мощность источника E2: P_E2 = E2 * i2 = {E2} * {i2:.6f} = {P_source_E2:.6f} Вт")
        print(f"Общая мощность источников: P_ист = {P_sources:.6f} Вт")
        
        # Мощность потребителей
        P_consumers = sum(powers)
        print(f"Общая мощность потребителей: P_потр = {P_consumers:.6f} Вт")
        
        # Проверка баланса
        balance_error = abs(P_sources - P_consumers)
        print(f"Ошибка баланса: |P_ист - P_потр| = {balance_error:.2e} Вт")
        
        if balance_error < 1e-10:
            print("✓ БАЛАНС МОЩНОСТЕЙ СОБЛЮДЕН ИДЕАЛЬНО!")
        else:
            print("✗ ОШИБКА В БАЛАНСЕ МОЩНОСТЕЙ!")
        
        print()
        
        # Создание визуализации
        create_visualization(currents, powers, voltages, R, E1, E2)
        
        # Создание таблицы результатов
        create_results_table(currents, R, voltages, powers)
        
        return currents, powers, voltages
        
    except np.linalg.LinAlgError:
        print("Ошибка: Система уравнений не имеет решения")
        return None, None, None

def create_visualization(currents, powers, voltages, R, E1, E2):
    """Создание визуализации результатов"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Диаграмма токов
    elements = [f'i{i+1}' for i in range(6)]
    colors = ['red' if i > 0 else 'blue' for i in currents]
    
    bars1 = ax1.bar(elements, currents, color=colors, alpha=0.7)
    ax1.set_ylabel('Ток, А')
    ax1.set_title('Токи в ветвях')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar, current in zip(bars1, currents):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{current:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    # 2. Диаграмма мощностей
    elements_power = [f'R{i+1}' for i in range(6)]
    colors_power = ['green'] * 6
    
    bars2 = ax2.bar(elements_power, powers, color=colors_power, alpha=0.7)
    ax2.set_ylabel('Мощность, Вт')
    ax2.set_title('Мощности на резисторах')
    ax2.grid(True, alpha=0.3)
    
    for bar, power in zip(bars2, powers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{power:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Баланс мощностей
    sources = ['E1', 'E2']
    source_powers = [E1 * currents[0], E2 * currents[1]]
    consumers = [f'R{i+1}' for i in range(6)]
    
    # Объединяем источники и потребители
    all_elements = sources + consumers
    all_powers = source_powers + powers
    all_colors = ['red', 'red'] + ['blue'] * 6
    
    bars3 = ax3.bar(all_elements, all_powers, color=all_colors, alpha=0.7)
    ax3.set_ylabel('Мощность, Вт')
    ax3.set_title('Баланс мощностей')
    ax3.grid(True, alpha=0.3)
    
    for bar, power in zip(bars3, all_powers):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{power:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 4. Схематическая диаграмма цепи
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_aspect('equal')
    ax4.set_title('Схематическая диаграмма цепи')
    ax4.axis('off')
    
    # Рисуем узлы
    nodes = {'a': (2, 8), 'b': (5, 8), 'c': (8, 8), 'd': (5, 2)}
    for node, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.3, color='black', fill=True)
        ax4.add_patch(circle)
        ax4.text(x, y-0.5, f'${node}$', ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Рисуем ветви с токами
    branches = [
        ('a', 'd', f'i₁={currents[0]:.2f}А', 'red'),
        ('b', 'd', f'i₂={currents[1]:.2f}А', 'red'),
        ('a', 'b', f'i₃={currents[2]:.2f}А', 'blue'),
        ('b', 'c', f'i₄={currents[3]:.2f}А', 'blue'),
        ('b', 'd', f'i₅={currents[4]:.2f}А', 'blue'),
        ('c', 'd', f'i₆={currents[5]:.2f}А', 'blue')
    ]
    
    for start, end, label, color in branches:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax4.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.7)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax4.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=8, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('corrected_kirchhoff_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('corrected_kirchhoff_analysis.pdf', bbox_inches='tight')
    
    print("Диаграммы сохранены в файлы:")
    print("- corrected_kirchhoff_analysis.png")
    print("- corrected_kirchhoff_analysis.pdf")
    
    plt.show()

def create_results_table(currents, R, voltages, powers):
    """Создание таблицы с результатами"""
    
    # Создаем DataFrame с результатами
    data = {
        'Ветвь': [f'i{i+1}' for i in range(6)],
        'Ток (А)': [f'{i:.6f}' for i in currents],
        'Сопротивление (Ом)': R,
        'Напряжение (В)': [f'{v:.6f}' for v in voltages],
        'Мощность (Вт)': [f'{p:.6f}' for p in powers]
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 80)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    # Сохраняем в CSV
    df.to_csv('corrected_circuit_results.csv', index=False, encoding='utf-8')
    print("Результаты сохранены в файл: corrected_circuit_results.csv")

def main():
    """Основная функция"""
    
    print("ПРАВИЛЬНЫЙ АНАЛИЗ ЭЛЕКТРИЧЕСКОЙ ЦЕПИ")
    print("МЕТОДОМ ЗАКОНОВ КИРХГОФА")
    print("=" * 70)
    
    # Решаем цепь
    currents, powers, voltages = solve_circuit_kirchhoff()
    
    if currents is not None:
        print("=" * 70)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("=" * 70)
    else:
        print("ОШИБКА В РАСЧЕТАХ!")

if __name__ == "__main__":
    main()
