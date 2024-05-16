from flask import Flask, request, abort, render_template_string
import threading
import time
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

# Хранилище для отслеживания IP и их активности
ip_activity = {}
blocked_ips = set()

# Пороговые значения
TIME_WINDOW = 60  # Временное окно в секундах

# Создание и обучение модели
X_train = np.array([
    [10, 0.5],
    [50, 0.1],
    [70, 0.05],
    [5, 1.0],
    [15, 0.7],
    [100, 0.02],
    [150, 0.01]
])
y_train = np.array([0, 1, 1, 0, 0, 1, 1])
model = LogisticRegression()
model.fit(X_train, y_train)


# Функция для очистки старых записей
def clean_up():
    while True:
        time.sleep(60)
        current_time = time.time()
        for ip in list(ip_activity.keys()):
            # Удаляем записи старше TIME_WINDOW
            ip_activity[ip] = [t for t in ip_activity[ip] if t > current_time - TIME_WINDOW]
            if not ip_activity[ip]:
                del ip_activity[ip]


# Запускаем поток для очистки
clean_up_thread = threading.Thread(target=clean_up)
clean_up_thread.daemon = True
clean_up_thread.start()


@app.route('/')
def index():
    return render_template_string('''
        <h1>Welcome to the protected site!</h1>
        <p>Enter some data below:</p>
        <form action="/submit" method="post">
            <input type="text" name="data" placeholder="Enter something">
            <input type="submit" value="Submit">
        </form>
        <h2>Blocked IPs</h2>
        <ul>
            {% for ip in blocked_ips %}
                <li>{{ ip }}</li>
            {% endfor %}
        </ul>
    ''', blocked_ips=blocked_ips)


@app.route('/submit', methods=['POST'])
def submit():
    ip = request.remote_addr
    current_time = time.time()

    if ip not in ip_activity:
        ip_activity[ip] = []

    ip_activity[ip] = [t for t in ip_activity[ip] if t > current_time - TIME_WINDOW]
    ip_activity[ip].append(current_time)

    num_requests = len(ip_activity[ip])
    avg_interval = (current_time - ip_activity[ip][0]) / num_requests if num_requests > 1 else TIME_WINDOW

    features = np.array([[num_requests, avg_interval]])
    prediction = model.predict(features)

    if prediction == 1:
        block_ip(ip)
        abort(403)  # Возвращаем ошибку доступа

    return 'Data received: ' + request.form['data']


def block_ip(ip):
    import subprocess
    subprocess.call(['sudo', 'iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'])
    blocked_ips.add(ip)
    print(f"Blocked IP: {ip}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
