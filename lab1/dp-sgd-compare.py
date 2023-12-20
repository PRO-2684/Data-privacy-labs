from importlib import import_module
from numpy import sqrt, log

dp = import_module("dp-sgd")
epsilons = (1, 10, 100, 500, 1000)
iters = (100, 500, 1000, 5000)
dataset_name = "cancer"
X_train, X_test, y_train, y_test = dp.get_train_data(dataset_name)


def test_model(epsilon, delta, iterations) -> float:
    """Test the model with given parameters and return the accuracy score."""
    dp_model = dp.LogisticRegressionCustom(
        learning_rate=0.01, num_iterations=iterations
    )
    dp_model.dp_fit(X_train, y_train, epsilon=epsilon, delta=delta, C=1)
    y_pred = dp_model.predict(X_test)
    accuracy = dp.accuracy_score(y_test, y_pred)
    return accuracy


def epsilon_u(epsilon, delta, epochs) -> float:
    """Calculate the epsilon_u."""
    delta_u = delta / (epochs + 1)
    epsilon_u = epsilon / (2 * sqrt(2 * epochs * log(1 / delta_u)))
    return epsilon_u


if __name__ == "__main__":
    print("Testing different epsilon with iteration=1000:")
    for epsilon in epsilons:
        print(
            f"  epsilon={epsilon}, delta=1, epsilon_u={epsilon_u(epsilon, 1, 1000):.3}: {test_model(epsilon, 1, 1000)}"
        )
    print("Testing different iterations with epsilon=100, delta=1:")
    for it in iters:
        print(f"  iterations={it}: {test_model(100, 1, it)}")
