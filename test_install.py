import sys
print(f"Python Version: {sys.version}")
print()

# Test each package
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except Exception as e:
    print(f"❌ Pandas: {e}")

try:
    from faker import Faker
    fake = Faker()
    print(f"✅ Faker: Installed (generated: {fake.name()})")
except Exception as e:
    print(f"❌ Faker: {e}")

try:
    import openpyxl
    print(f"✅ openpyxl: {openpyxl.__version__}")
except Exception as e:
    print(f"❌ openpyxl: {e}")

# Test basic functionality
print("\n🧪 Testing data generation...")
fake = Faker()
data = {
    'name': [fake.name() for _ in range(3)],
    'email': [fake.email() for _ in range(3)],
    'balance': np.random.normal(10000, 2000, 3).round(2)
}
df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)
print(f"\n✅ All tests passed! You're ready to build banking data.")