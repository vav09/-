# Постоянная ссылка: публикация на Streamlit Community Cloud

Бесплатно. Получится адрес вида `https://ваше-имя.streamlit.app`, который открывается
с любого компьютера и телефона, без запуска программы у вас.

Нужно: аккаунт GitHub (есть) — около 10 минут.

## 1. Выложить код на GitHub (без командной строки)

1. Зайдите на https://github.com, справа вверху **+** → **New repository**.
2. Repository name: например `pkps-wave-moment`. Видимость — **Public**
   (для Private тоже работает, но придётся дать Streamlit доступ к приватным репозиториям).
   Остальное не трогайте → **Create repository**.
3. На странице пустого репозитория нажмите ссылку **uploading an existing file**.
4. Распакуйте архив `pkps_wave_moment.zip` и перетащите в окно браузера **всё содержимое папки**
   `pkps_wave_moment` (не саму папку): `app.py`, `requirements.txt`, все `pkps_*.py`, папку `assets`.
5. ⚠️ Папка `.streamlit` начинается с точки — в Windows и macOS она может быть скрыта и не перетащиться.
   Проверьте после загрузки, что в репозитории есть `.streamlit/config.toml`. Если нет:
   **Add file → Create new file**, имя `.streamlit/config.toml` (с косой чертой — GitHub сам создаст папку),
   вставьте содержимое файла из архива → **Commit changes**.
   Без него приложение тоже работает, только тема будет светлая вместо тёмной.
6. Внизу **Commit changes**.

Через git это делается так (если удобнее):
```
cd pkps_wave_moment
git init
git add .
git commit -m "Волновой момент ПКПС РКО"
git branch -M main
git remote add origin https://github.com/<ваш-логин>/pkps-wave-moment.git
git push -u origin main
```

## 2. Опубликовать

1. Зайдите на https://share.streamlit.io и войдите через **Continue with GitHub** (разрешите доступ).
2. **Create app** → **Deploy a public app from GitHub** (или «Yup, I have an app»).
3. Repository: `<ваш-логин>/pkps-wave-moment`, Branch: `main`, Main file path: `app.py`.
4. **App URL** — придумайте адрес, например `pkps-volna` → будет `https://pkps-volna.streamlit.app`.
5. В **Advanced settings** выберите Python **3.12** → **Deploy**.

Первый запуск занимает 2–5 минут (ставятся библиотеки из `requirements.txt`). Дальше ссылку можно
отправлять кому угодно.

## 3. Как обновлять

Замените файлы в репозитории на GitHub (**Add file → Upload files** с теми же именами) —
приложение пересоберётся само через минуту.

## Что важно знать

- **Проекты пользователей на сервере не хранятся.** Каждый сохраняет свой проект кнопкой
  «Сохранить проект» себе на компьютер и открывает его снова кнопкой «Открыть проект».
  Чертежи PDF тоже никуда не сохраняются — они живут только в открытой вкладке браузера.
- Если приложением неделю никто не пользуется, Streamlit его «усыпляет»: первый посетитель увидит
  кнопку **Yes, get this app back up!** и подождёт около минуты.
- Публичное приложение доступно всем, у кого есть ссылка. Чтобы пускать только своих:
  в настройках приложения на share.streamlit.io → **Sharing** → «Only specific people can view this app»
  и впишите их e-mail.
- Лимиты бесплатного тарифа (около 1 ГБ памяти) для этого приложения с запасом.
