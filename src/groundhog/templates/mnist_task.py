# /// script
# dependencies = ["groundhog-researcher", "python-dotenv", "numpy", "scikit-learn", "torch"]
# ///
"""MNIST classification task for groundhog.

Real task: classify handwritten digits using only 50 training samples.
Time limit passed to code so it can adapt. External timeout via subprocess.

Five evaluation stages:
    smoke (instant)  → validate (15s, 1k)  → evaluate (60s, 10k)
    → deep (10min, 10k) → full (100min, 10k)

During optimization, run through="evaluate". Deep and full are for final validation.
"""

import traceback

import numpy as np

from groundhog import Task, Data, Context, Evaluator, EvalStage, StageResult, run_code


class MNISTData(Data):
    def __init__(self, samples_per_digit: int = 5):
        self.samples_per_digit = samples_per_digit
        self._load()

    def _load(self):
        from sklearn.datasets import fetch_openml
        import sys
        print("Loading MNIST...", file=sys.stderr)
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)

        X_train_full, X_test = X[:60000], X[60000:]
        y_train_full, y_test = y[:60000], y[60000:]

        X_train, y_train = [], []
        for digit in range(10):
            mask = y_train_full == digit
            X_train.append(X_train_full[mask][:self.samples_per_digit])
            y_train.extend([digit] * self.samples_per_digit)

        self._train = (np.vstack(X_train), np.array(y_train))
        self._test = (X_test, y_test)
        print(f"Train: {self._train[0].shape}, Test: {self._test[0].shape}", file=sys.stderr)

    def get_train(self):
        return self._train

    def get_test(self):
        return self._test


class MNISTContext(Context):
    def get_brief(self):
        return (
            "Optimize a digit classifier for MNIST. "
            "Goal: maximize accuracy on 10k test images using only 50 training samples. "
            "You have 60 seconds of compute budget."
        )

    def get_extended(self):
        return """Write a Python function `run(train_data, time_limit)` that returns a predictor for MNIST digits.

Input:
- train_data: tuple (X_train, y_train)
  - X_train: numpy array shape (50, 784) — 5 images per digit, normalized 0-1
  - y_train: numpy array shape (50,) — labels 0-9
- time_limit: float — seconds of compute budget available for training.
  Your algorithm should use this to scale its training effort, but always use same method just for differnt durations.
  Target budget is 60 seconds.

Output: A callable `predict(X)` that takes images (N, 784) and returns predictions (integers 0-9)

The challenge: with only 50 labeled samples (5 per digit), you need creative
approaches to generalize. Consider data augmentation, distance metrics,
prototype-based methods, or any technique that works well with very few samples.

Example:
    def run(train_data, time_limit):
        X_train, y_train = train_data
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        def predict(X):
            return model.predict(X)
        return predict

Available in namespace: numpy as np, torch, sklearn
No other imports allowed."""


class MNISTEvaluator(Evaluator):

    IMPORTS = {"np": "numpy", "sklearn": "sklearn", "torch": "torch"}

    def _evaluate_on(self, code, data, max_samples, time_limit):
        """Run code in subprocess with timeout, score predictions."""
        X_test, y_test = data.get_test()
        if max_samples and len(X_test) > max_samples:
            idx = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]

        # Wrapper that runs user code and returns predictions
        wrapper = f'''
{code}

def _evaluate(train_data, X_test, time_limit):
    import time
    start = time.time()
    predictor = run(train_data, time_limit)
    predictions = np.array(predictor(X_test)).astype(int)
    elapsed = time.time() - start
    return predictions, elapsed
'''
        try:
            predictions, elapsed = run_code(
                code=wrapper,
                entry_point="_evaluate",
                args=(data.get_train(), X_test, time_limit),
                imports=self.IMPORTS,
                timeout=int(time_limit * 1.5),
            )
        except TimeoutError:
            return StageResult(errors={"timeout": f"Exceeded {time_limit}s"})
        except RuntimeError as e:
            return StageResult(errors={"runtime": str(e)})

        accuracy = float(np.mean(predictions == y_test))
        per_class = {}
        for digit in range(10):
            mask = y_test == digit
            if mask.sum() > 0:
                per_class[f"class_{digit}"] = float(np.mean(predictions[mask] == digit))

        return StageResult(
            score=accuracy,
            metrics={
                "accuracy": accuracy,
                "exec_time": elapsed,
                "time_limit": time_limit,
                "n_samples": len(y_test),
                **per_class,
            },
        )

    def evaluate(self, code, data):
        return self._evaluate_on(code, data, max_samples=None, time_limit=60)

    @staticmethod
    def _accuracy_scorer(result):
        if result.errors:
            return -1.0
        return result.metrics.get("accuracy", 0.0)

    @staticmethod
    def _smoke_scorer(result):
        return -1.0 if result.errors else 1.0

    def get_stages(self, data):
        return [
            EvalStage("smoke", "Compiles and defines run()",
                      lambda code: self._smoke(code),
                      scorer=self._smoke_scorer),
            EvalStage("validate", "1k samples, 15s",
                      lambda code, d=data: self._evaluate_on(code, d, max_samples=1000, time_limit=15),
                      scorer=self._accuracy_scorer),
            EvalStage("evaluate", "10k samples, 60s",
                      lambda code, d=data: self._evaluate_on(code, d, max_samples=None, time_limit=60),
                      scorer=self._accuracy_scorer),
            EvalStage("deep", "10k samples, 10min",
                      lambda code, d=data: self._evaluate_on(code, d, max_samples=None, time_limit=600),
                      scorer=self._accuracy_scorer),
            EvalStage("full", "10k samples, 100min",
                      lambda code, d=data: self._evaluate_on(code, d, max_samples=None, time_limit=6000),
                      scorer=self._accuracy_scorer),
        ]

    def _smoke(self, code):
        try:
            ns = {}
            exec(code, ns)
            if "run" not in ns:
                return StageResult(errors={"missing": "No run() function defined"})
            if not callable(ns["run"]):
                return StageResult(errors={"type": "run is not callable"})
            return StageResult(score=1.0, metrics={"compiles": 1.0})
        except Exception as e:
            return StageResult(errors={"syntax": str(e)})


class MNISTTask(Task):
    def __init__(self):
        super().__init__(
            data=MNISTData(),
            context=MNISTContext(),
            evaluator=MNISTEvaluator(),
            name="MNIST",
        )


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    from groundhog import (
        SimpleOptimizer, Improve, FreshApproach, CrossPollinate, Analyse,
        GeminiBackend, BackendRegistry,
    )

    task = MNISTTask()

    optimizer = SimpleOptimizer(
        task,
        strategies=[
            (FreshApproach(mode="different"), 1),
            (Improve(), 7),
            (CrossPollinate(), 2),
            (Improve(), 7),
            (CrossPollinate(), 3),
        ],
        seed_strategy=FreshApproach(mode="blank"),
        through="evaluate",
    )
    optimizer.toolkit.llm = BackendRegistry(
        max=GeminiBackend(model="gemini-3.1-pro-preview"),
        high=GeminiBackend(model="gemini-3-flash-preview"),
        default=GeminiBackend(model="gemini-3.1-flash-lite-preview"),
        budget=GeminiBackend(model="gemini-3.1-flash-lite-preview", thinking_level="MINIMAL"),
        cheap=GeminiBackend(model="gemini-2.5-flash-lite"),
    )

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        optimizer.status()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        optimizer.run(n=n)
