from importlib import import_module

ROUNDS = 16
dp = import_module("dp-sgd")
dp_args = (
    (0.1, 1e-2),
    (0.1, 1e-3),
    (0.1, 1e-4),
    (1.0, 1e-2),
    (1.0, 1e-3),
    (1.0, 1e-4),
    (10.0, 1e-2),
    (10.0, 1e-3),
    (10.0, 1e-4),
)
iters = 100, 500, 1000, 5000
dataset_name = "cancer"
X_train, X_test, y_train, y_test = dp.get_train_data(dataset_name)


def test_model(epsilon, delta, iterations) -> float:
    """Test the model with given parameters and return the accuracy score."""
    sumAcc = 0
    for _ in range(ROUNDS):
        dp_model = dp.LogisticRegressionCustom(
            learning_rate=0.01, num_iterations=iterations
        )
        dp_model.dp_fit(X_train, y_train, epsilon=epsilon, delta=delta, C=1)
        y_pred = dp_model.predict(X_test)
        accuracy = dp.accuracy_score(y_test, y_pred)
        sumAcc += accuracy
    return sumAcc / ROUNDS


if __name__ == "__main__":
    print("Testing different epsilon, delta with iteration=1000:")
    for epsilon, delta in dp_args:
        print(f"  epsilon={epsilon}, delta={delta}: {test_model(epsilon, delta, 1000)}")
    print("Testing different iterations with epsilon=1.0, delta=1e-3:")
    for it in iters:
        print(f"  iterations={it}: {test_model(1.0, 1e-3, it)}")
