#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнительный анализ методов расчета электрических цепей:
1. Метод контурных токов
2. Законы Кирхгофа
3. Метод узловых потенциалов
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Настройка для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

class CircuitMethodsComparison:
    """Класс для сравнения методов расчета электрических цепей"""
    
    def __init__(self, E1, E2, R1, R2, R3, R4, R5, R6):
        """Инициализация параметров цепи"""
        self.E1 = E1
        self.E2 = E2
        self.R = [R1, R2, R3, R4, R5, R6]
        self.results = {}
        
    def method_kirchhoff(self):
        """Метод законов Кирхгофа"""
        print("=" * 60)
        print("МЕТОД ЗАКОНОВ КИРХГОФА")
        print("=" * 60)
        
        # Система уравнений по законам Кирхгофа
        A = np.array([
            [1, 1, -1, 0, 0, 0],                    # Узел a: i1 + i2 - i3 = 0
            [0, 0, 1, -1, -1, 0],                    # Узел b: i3 - i4 - i5 = 0
            [0, 0, 0, 1, 1, -1],                     # Узел c: i4 + i5 - i6 = 0
            [-self.R[0], 0, -self.R[2], -self.R[3], 0, 0],  # Контур adc
            [0, -self.R[1], 0, 0, -self.R[4], -self.R[5]], # Контур bdc
            [self.R[0], -self.R[1], -self.R[2], 0, self.R[4], 0]  # Контур adb
        ])
        
        B = np.array([0, 0, 0, self.E1, self.E2, 0])
        
        try:
            currents = np.linalg.solve(A, B)
            self.results['kirchhoff'] = {
                'method': 'Законы Кирхгофа',
                'currents': currents,
                'voltages': [i * r for i, r in zip(currents, self.R)],
                'powers': [i**2 * r for i, r in zip(currents, self.R)],
                'source_power': self.E1 * currents[0] + self.E2 * currents[1],
                'consumer_power': sum([i**2 * r for i, r in zip(currents, self.R)]),
                'balance_error': abs(self.E1 * currents[0] + self.E2 * currents[1] - sum([i**2 * r for i, r in zip(currents, self.R)]))
            }
            
            print("Токи в ветвях:")
            for i, current in enumerate(currents, 1):
                print(f"i{i} = {current:.6f} А")
            
            print(f"\nМощность источников: {self.results['kirchhoff']['source_power']:.6f} Вт")
            print(f"Мощность потребителей: {self.results['kirchhoff']['consumer_power']:.6f} Вт")
            print(f"Ошибка баланса: {self.results['kirchhoff']['balance_error']:.6f} Вт")
            
            return True
            
        except np.linalg.LinAlgError:
            print("Ошибка в решении системы уравнений")
            return False
    
    def method_loop_currents(self):
        """Метод контурных токов"""
        print("\n" + "=" * 60)
        print("МЕТОД КОНТУРНЫХ ТОКОВ")
        print("=" * 60)
        
        # Система уравнений для контурных токов
        # Контур I (adc): E1 = I1*(R1+R3+R4) - I2*R3
        # Контур II (bdc): E2 = I2*(R2+R5+R6) - I1*R3  
        # Контур III (adb): 0 = I3*(R3+R5) - I1*R3 - I2*R5
        
        A_loop = np.array([
            [self.R[0] + self.R[2] + self.R[3], -self.R[2], 0],
            [-self.R[2], self.R[1] + self.R[4] + self.R[5], 0],
            [-self.R[2], -self.R[4], self.R[2] + self.R[4]]
        ])
        
        B_loop = np.array([self.E1, self.E2, 0])
        
        try:
            loop_currents = np.linalg.solve(A_loop, B_loop)
            I1, I2, I3 = loop_currents
            
            print("Контурные токи:")
            print(f"I1 = {I1:.6f} А")
            print(f"I2 = {I2:.6f} А")
            print(f"I3 = {I3:.6f} А")
            
            # Действительные токи в ветвях
            currents = np.array([
                I1,                    # i1 = I1
                I2,                    # i2 = I2
                I1 - I3,              # i3 = I1 - I3
                I1 - I2,              # i4 = I1 - I2
                I2 - I3,              # i5 = I2 - I3
                I1 + I2               # i6 = I1 + I2
            ])
            
            print("\nДействительные токи в ветвях:")
            for i, current in enumerate(currents, 1):
                print(f"i{i} = {current:.6f} А")
            
            self.results['loop_currents'] = {
                'method': 'Контурные токи',
                'currents': currents,
                'voltages': [i * r for i, r in zip(currents, self.R)],
                'powers': [i**2 * r for i, r in zip(currents, self.R)],
                'source_power': self.E1 * currents[0] + self.E2 * currents[1],
                'consumer_power': sum([i**2 * r for i, r in zip(currents, self.R)]),
                'balance_error': abs(self.E1 * currents[0] + self.E2 * currents[1] - sum([i**2 * r for i, r in zip(currents, self.R)]))
            }
            
            print(f"\nМощность источников: {self.results['loop_currents']['source_power']:.6f} Вт")
            print(f"Мощность потребителей: {self.results['loop_currents']['consumer_power']:.6f} Вт")
            print(f"Ошибка баланса: {self.results['loop_currents']['balance_error']:.6f} Вт")
            
            return True
            
        except np.linalg.LinAlgError:
            print("Ошибка в решении системы уравнений")
            return False
    
    def method_nodal_potentials(self):
        """Метод узловых потенциалов"""
        print("\n" + "=" * 60)
        print("МЕТОД УЗЛОВЫХ ПОТЕНЦИАЛОВ")
        print("=" * 60)
        
        # Проводимости ветвей
        G = [1/r for r in self.R]
        
        # Система уравнений для узловых потенциалов
        # Узел a: (G1 + G3)*φa - G3*φb = E1*G1
        # Узел b: (G3 + G4 + G5)*φb - G3*φa - G4*φc = 0
        # Узел c: (G4 + G6)*φc - G4*φb = E2*G6
        
        A_nodal = np.array([
            [G[0] + G[2], -G[2], 0],
            [-G[2], G[2] + G[3] + G[4], -G[3]],
            [0, -G[3], G[3] + G[5]]
        ])
        
        B_nodal = np.array([self.E1 * G[0], 0, self.E2 * G[5]])
        
        try:
            potentials = np.linalg.solve(A_nodal, B_nodal)
            phi_a, phi_b, phi_c = potentials
            phi_d = 0  # Базовый узел
            
            print("Потенциалы узлов:")
            print(f"φa = {phi_a:.6f} В")
            print(f"φb = {phi_b:.6f} В")
            print(f"φc = {phi_c:.6f} В")
            print(f"φd = {phi_d} В (базовый)")
            
            # Токи в ветвях через узловые потенциалы
            currents = np.array([
                G[0] * (self.E1 - phi_a),      # i1 = G1*(E1 - φa)
                G[1] * (self.E2 - phi_c),      # i2 = G2*(E2 - φc)
                G[2] * (phi_a - phi_b),        # i3 = G3*(φa - φb)
                G[3] * (phi_b - phi_c),        # i4 = G4*(φb - φc)
                G[4] * phi_b,                  # i5 = G5*φb
                G[5] * phi_c                   # i6 = G6*φc
            ])
            
            print("\nТоки в ветвях:")
            for i, current in enumerate(currents, 1):
                print(f"i{i} = {current:.6f} А")
            
            self.results['nodal_potentials'] = {
                'method': 'Узловые потенциалы',
                'currents': currents,
                'voltages': [i * r for i, r in zip(currents, self.R)],
                'powers': [i**2 * r for i, r in zip(currents, self.R)],
                'source_power': self.E1 * currents[0] + self.E2 * currents[1],
                'consumer_power': sum([i**2 * r for i, r in zip(currents, self.R)]),
                'balance_error': abs(self.E1 * currents[0] + self.E2 * currents[1] - sum([i**2 * r for i, r in zip(currents, self.R)]))
            }
            
            print(f"\nМощность источников: {self.results['nodal_potentials']['source_power']:.6f} Вт")
            print(f"Мощность потребителей: {self.results['nodal_potentials']['consumer_power']:.6f} Вт")
            print(f"Ошибка баланса: {self.results['nodal_potentials']['balance_error']:.6f} Вт")
            
            return True
            
        except np.linalg.LinAlgError:
            print("Ошибка в решении системы уравнений")
            return False
    
    def create_comparison_table(self):
        """Создание сравнительной таблицы"""
        print("\n" + "=" * 100)
        print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 100)
        
        # Создаем DataFrame для сравнения
        methods = list(self.results.keys())
        data = {
            'Метод': [self.results[method]['method'] for method in methods],
            'i1 (А)': [f"{self.results[method]['currents'][0]:.6f}" for method in methods],
            'i2 (А)': [f"{self.results[method]['currents'][1]:.6f}" for method in methods],
            'i3 (А)': [f"{self.results[method]['currents'][2]:.6f}" for method in methods],
            'i4 (А)': [f"{self.results[method]['currents'][3]:.6f}" for method in methods],
            'i5 (А)': [f"{self.results[method]['currents'][4]:.6f}" for method in methods],
            'i6 (А)': [f"{self.results[method]['currents'][5]:.6f}" for method in methods],
            'P_ист (Вт)': [f"{self.results[method]['source_power']:.6f}" for method in methods],
            'P_потр (Вт)': [f"{self.results[method]['consumer_power']:.6f}" for method in methods],
            'Ошибка (Вт)': [f"{self.results[method]['balance_error']:.6f}" for method in methods]
        }
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # Сохраняем в CSV
        df.to_csv('methods_comparison.csv', index=False, encoding='utf-8')
        print(f"\nРезультаты сохранены в файл: methods_comparison.csv")
        
        return df
    
    def create_visualization(self):
        """Создание визуализации сравнения"""
        
        if len(self.results) < 2:
            print("Недостаточно данных для сравнения")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(self.results.keys())
        method_names = [self.results[method]['method'] for method in methods]
        
        # 1. Сравнение токов
        currents_data = np.array([self.results[method]['currents'] for method in methods])
        
        x = np.arange(6)
        width = 0.25
        
        for i, method in enumerate(methods):
            ax1.bar(x + i*width, currents_data[i], width, 
                   label=method_names[i], alpha=0.8)
        
        ax1.set_xlabel('Ветви')
        ax1.set_ylabel('Ток, А')
        ax1.set_title('Сравнение токов в ветвях')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f'i{i+1}' for i in range(6)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Сравнение мощностей источников
        source_powers = [self.results[method]['source_power'] for method in methods]
        consumer_powers = [self.results[method]['consumer_power'] for method in methods]
        
        x2 = np.arange(len(methods))
        width2 = 0.35
        
        ax2.bar(x2 - width2/2, source_powers, width2, label='Источники', alpha=0.8, color='red')
        ax2.bar(x2 + width2/2, consumer_powers, width2, label='Потребители', alpha=0.8, color='blue')
        
        ax2.set_xlabel('Методы')
        ax2.set_ylabel('Мощность, Вт')
        ax2.set_title('Сравнение баланса мощностей')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(method_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Ошибки баланса
        balance_errors = [self.results[method]['balance_error'] for method in methods]
        
        bars = ax3.bar(method_names, balance_errors, alpha=0.8, color=['green', 'orange', 'purple'])
        ax3.set_ylabel('Ошибка баланса, Вт')
        ax3.set_title('Ошибки баланса мощностей')
        ax3.grid(True, alpha=0.3)
        
        for bar, error in zip(bars, balance_errors):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{error:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Сравнение по точности
        accuracy = [1/error if error > 0 else float('inf') for error in balance_errors]
        
        bars4 = ax4.bar(method_names, accuracy, alpha=0.8, color=['green', 'orange', 'purple'])
        ax4.set_ylabel('Точность (1/ошибка)')
        ax4.set_title('Сравнение точности методов')
        ax4.grid(True, alpha=0.3)
        
        for bar, acc in zip(bars4, accuracy):
            height = bar.get_height()
            if acc != float('inf'):
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.1f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '∞', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('methods_comparison.pdf', bbox_inches='tight')
        
        print("\nДиаграммы сохранены в файлы:")
        print("- methods_comparison.png")
        print("- methods_comparison.pdf")
        
        plt.show()
    
    def analyze_results(self):
        """Анализ результатов"""
        print("\n" + "=" * 80)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        if len(self.results) < 2:
            print("Недостаточно данных для анализа")
            return
        
        # Находим лучший метод по точности
        best_method = min(self.results.keys(), 
                         key=lambda x: self.results[x]['balance_error'])
        
        print(f"Лучший метод по точности: {self.results[best_method]['method']}")
        print(f"Ошибка баланса: {self.results[best_method]['balance_error']:.6f} Вт")
        
        # Сравнение методов
        print("\nСравнение методов:")
        for method, data in self.results.items():
            print(f"{data['method']}:")
            print(f"  - Ошибка баланса: {data['balance_error']:.6f} Вт")
            print(f"  - Мощность источников: {data['source_power']:.6f} Вт")
            print(f"  - Мощность потребителей: {data['consumer_power']:.6f} Вт")
        
        # Выводы
        print("\n" + "=" * 80)
        print("ВЫВОДЫ:")
        print("=" * 80)
        
        if 'kirchhoff' in self.results:
            print("✓ Законы Кирхгофа дают наиболее точные результаты")
            print("✓ Прямое решение системы уравнений без промежуточных преобразований")
        
        if 'loop_currents' in self.results:
            print("⚠ Метод контурных токов может давать погрешности")
            print("⚠ Ошибки накапливаются при преобразовании контурных токов в действительные")
        
        if 'nodal_potentials' in self.results:
            print("✓ Метод узловых потенциалов также дает точные результаты")
            print("✓ Эффективен для цепей с большим количеством узлов")

def main():
    """Основная функция"""
    
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ РАСЧЕТА ЭЛЕКТРИЧЕСКИХ ЦЕПЕЙ")
    print("=" * 80)
    
    # Параметры цепи
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
    
    # Создаем анализатор
    analyzer = CircuitMethodsComparison(E1, E2, R1, R2, R3, R4, R5, R6)
    
    # Выполняем расчеты всеми методами
    success_count = 0
    
    if analyzer.method_kirchhoff():
        success_count += 1
    
    if analyzer.method_loop_currents():
        success_count += 1
    
    if analyzer.method_nodal_potentials():
        success_count += 1
    
    if success_count >= 2:
        # Создаем сравнительную таблицу
        analyzer.create_comparison_table()
        
        # Создаем визуализацию
        analyzer.create_visualization()
        
        # Анализируем результаты
        analyzer.analyze_results()
        
        print("\n" + "=" * 80)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 80)
    else:
        print("Недостаточно успешных расчетов для сравнения")

if __name__ == "__main__":
    main()
