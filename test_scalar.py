from metanion import Metanion, Tensor
import numpy as np

print("=" * 70)
print("TEST: 0D (SCALAR) OPERATIONS")
print("=" * 70)

# TEST 1: SCALAR CREATION
print("\n[1] SCALAR CREATION")
try:
    t1 = Tensor(5.0)
    print(f"  Tensor(5.0) = {t1}")
    print(f"  Shape: {t1.shape}")
    print(f"  Dtype: {t1.dtype}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# TEST 2: SCALAR ADDITION
print("\n[2] SCALAR ADDITION")
try:
    t1 = Tensor(5.0)
    t2 = Tensor(3.0)
    t3 = t1 + t2
    print(f"  {t1} + {t2} = {t3}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# TEST 3: SCALAR SUBTRACTION
print("\n[3] SCALAR SUBTRACTION")
try:
    t1 = Tensor(5.0)
    t2 = Tensor(3.0)
    t3 = t1 - t2
    print(f"  {t1} - {t2} = {t3}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# TEST 4: SCALAR MULTIPLICATION
print("\n[4] SCALAR MULTIPLICATION")
try:
    t1 = Tensor(5.0)
    t2 = Tensor(3.0)
    t3 = t1 * t2
    print(f"  {t1} * {t2} = {t3}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# TEST 5: SCALAR DIVISION
print("\n[5] SCALAR DIVISION")
try:
    t1 = Tensor(6.0)
    t2 = Tensor(2.0)
    t3 = t1 / t2
    print(f"  {t1} / {t2} = {t3}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# TEST 6: SCALAR TO LIST
print("\n[6] SCALAR TO LIST")
try:
    t1 = Tensor(5.0)
    lst = t1.to_list()
    print(f"  {t1}.to_list() = {lst}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# TEST 7: SCALAR WITH METANION MODEL
print("\n[7] SCALAR WITH METANION MODEL")
try:
    X = np.random.randn(20, 1)
    y = 2*X[:,0] + 1 + 0.1*np.random.randn(20)
    model = Metanion(verbose=False, random_seed=42)
    model.fit(X, y, feature_names=["x"])
    pred = model.predict(np.array([[2.0]]))
    print(f"  Model prediction for x=2: {pred[0]:.4f}")
    print(f"  Equation: {model.explain()}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
