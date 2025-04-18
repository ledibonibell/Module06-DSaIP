import numpy as np


# Функции активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Инициализация параметров
w = np.array([0.45, 0.78, -0.12, 0.13, 1.5, -2.3], dtype=float)
E = 0.7
A = 0.3
I = np.array([1, 0])
target = 1


# Форматирование вывода
def print_header(text):
    print(f"\n{'=' * 40}")
    print(f"{text:^40}")
    print(f"{'=' * 40}")


def print_subheader(text):
    print(f"\n{'—' * 30}")
    print(f"{text}")
    print(f"{'—' * 30}")


for epoch in range(1):
    print_header(f"ЭПОХА {epoch + 1}")

    # ======================================
    # ПРЯМОЙ ХОД
    # ======================================
    print_subheader("ПРЯМОЙ ХОД")

    # Вычисления
    # Скрытый слой
    h1_input = w[0] * I[0] + w[2] * I[1]
    h1 = sigmoid(h1_input)

    h2_input = w[1] * I[0] + w[3] * I[1]
    h2 = sigmoid(h2_input)

    # Выходной слой
    output_input = w[4] * h1 + w[5] * h2
    output = sigmoid(output_input)
    MSE = (target - output) ** 2
    error = (target - output)

    # Вывод результатов
    print(f"H1: {w[0]:.4f} * {I[0]} + {w[1]:.4f} * {I[1]} = {h1_input:.4f}")
    print(f"    sigmoid({h1_input:.4f}) = {h1:.4f}")
    print(f"H2: {w[2]:.4f} * {I[0]} + {w[3]:.4f} * {I[1]} = {h2_input:.4f}")
    print(f"    sigmoid({h2_input:.4f}) = {h2:.4f}")

    print(f"\nВыходной слой:")
    print(f"O1: {w[4]:.4f} * {h1:.4f} + {w[5]:.4f} * {h2:.4f} = {output_input:.4f}")
    print(f"    sigmoid({output_input:.4f}) = {output:.4f}")

    print(f"\nРезультат:")
    print(f"Вычисленный выход: {output:.4f}")
    print(f"Ожидаемый выход:   {target}")
    print(f"Ошибка:           {error:.4f}")

    # ======================================
    # ОБРАТНЫЙ ХОД
    # ======================================
    print_subheader("ОБРАТНЫЙ ХОД")

    # Вычисления
    # Дельта выходного слоя
    delta_output = error * sigmoid_derivative(output)

    # Обновление весов выходного слоя
    delta_w5 = E * delta_output * h1
    delta_w6 = E * delta_output * h2

    # Дельта скрытого слоя
    delta_h1 = delta_output * w[4] * sigmoid_derivative(h1)
    delta_h2 = delta_output * w[5] * sigmoid_derivative(h2)

    # Обновление весов скрытого слоя
    delta_w1 = E * delta_h1 * I[0]
    delta_w2 = E * delta_h1 * I[1]
    delta_w3 = E * delta_h2 * I[0]
    delta_w4 = E * delta_h2 * I[1]

    # Применение изменений весов
    new_w = w.copy()
    new_w[0] += delta_w1
    new_w[1] += delta_w2
    new_w[2] += delta_w3
    new_w[3] += delta_w4
    new_w[4] += delta_w5
    new_w[5] += delta_w6

    # Вывод результатов
    print(f"\nДельта выходного слоя:")
    print(f"δ_output = error * output*(1-output) = {error:.4f} * {output:.4f}*(1-{output:.4f}) = {delta_output:.4f}")

    print(f"\nОбновление весов выходного слоя:")
    print(f"Δw5 = η * δ_output * h1 = {E} * {delta_output:.4f} * {h1:.4f} = {delta_w5:.4f}")
    print(f"Δw6 = η * δ_output * h2 = {E} * {delta_output:.4f} * {h2:.4f} = {delta_w6:.4f}")

    print(f"\nДельта скрытого слоя:")
    print(
        f"δ_h1 = δ_output * w5 * h1*(1-h1) = {delta_output:.4f} * {w[4]:.4f} * {h1:.4f}*(1-{h1:.4f}) = {delta_h1:.4f}")
    print(
        f"δ_h2 = δ_output * w6 * h2*(1-h2) = {delta_output:.4f} * {w[5]:.4f} * {h2:.4f}*(1-{h2:.4f}) = {delta_h2:.4f}")

    print(f"\nОбновление весов скрытого слоя:")
    print(f"Δw1 = η * δ_h1 * I1 = {E} * {delta_h1:.4f} * {I[0]} = {delta_w1:.4f}")
    print(f"Δw2 = η * δ_h1 * I2 = {E} * {delta_h1:.4f} * {I[1]} = {delta_w2:.4f}")
    print(f"Δw3 = η * δ_h2 * I1 = {E} * {delta_h2:.4f} * {I[0]} = {delta_w3:.4f}")
    print(f"Δw4 = η * δ_h2 * I2 = {E} * {delta_h2:.4f} * {I[1]} = {delta_w4:.4f}")

    print(f"\nНовые веса:")
    for i, (old, new) in enumerate(zip(w, new_w), 1):
        print(f"w{i}: {old:.4f} → {new:.4f} (Δ = {new - old:+.4f})")

    w = new_w

print_header("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("\nИтоговые веса:")
for i, weight in enumerate(w, 1):
    print(f"w{i} = {weight:.4f}")

# ======================================
# ФИНАЛЬНЫЙ РАСЧЕТ С ИТОГОВЫМИ ВЕСАМИ
# ======================================
print_subheader("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")

# Вычисляем результат с обновленными весами
h1_final = sigmoid(w[0] * I[0] + w[2] * I[1])
h2_final = sigmoid(w[1] * I[0] + w[3] * I[1])
output_final = sigmoid(w[4] * h1_final + w[5] * h2_final)

print(f"\nС вычисленными весами:")
print(f"H1 = {w[0]:.4f} * {I[0]} + {w[2]:.4f} * {I[1]} = {w[0] * I[0] + w[2] * I[1]:.4f}")
print(f"    sigmoid({w[0] * I[0] + w[2] * I[1]:.4f}) = {h1_final:.4f}")
print(f"H2 = {w[1]:.4f} * {I[0]} + {w[3]:.4f} * {I[1]} = {w[1] * I[0] + w[3] * I[1]:.4f}")
print(f"    sigmoid({w[1] * I[0] + w[3] * I[1]:.4f}) = {h2_final:.4f}")

print(f"\nO1 = {w[4]:.4f} * {h1_final:.4f} + {w[5]:.4f} * {h2_final:.4f} = {w[4] * h1_final + w[5] * h2_final:.4f}")
print(f"    sigmoid({w[4] * h1_final + w[5] * h2_final:.4f}) = {output_final:.4f}")

print(f"\nИтоговый результат:")
print(f"Y (вычисленный): {output_final:.4f}")
print(f"Y (ожидаемый):   {target}")
print(f"Ошибка:         {target - output_final:.4f}")