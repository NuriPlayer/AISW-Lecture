import tensorflow as tf

# 학습 데이터 집어넣기
X_data = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
Y_data = tf.constant([10, 20, 30, 40, 50], dtype=tf.float32)

# 가중치와 편향치 만들어줌
w = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

# 예측값에 따라 손실함수 만들어주기
def loss_function():
    # 예측값: 1차 함수 형태
    Y_predict = w * X_data + b
    # 평균제곱오차
    return tf.reduce_mean(tf.square(Y_data - Y_predict))

# Adam 옵티마이저: 성능 좋은 편이니 일단 한번 써봅시다
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

# 손실함수를 넣어주고 w, b 만번 업데이트
for step in range(10000):
    optimizer.minimize(loss_function, var_list=[w, b]) 
    print(step, w, b)
