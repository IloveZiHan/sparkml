
import pandas as pd

df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)

age_series = df["Age"]

print(type(age_series))
print(age_series)

print("-" * 80)

ages = pd.Series([22, 35, 58], name="Age")
print(ages)

import string

class Car:
    def __init__(self, brand: string, color: string):
        self.brand = brand
        self.color = color

    def hello(self):
        print(f"Hello, {self.brand}!")


class BenZi(Car):
    def __init__(self, brand: string, color:string):
        super().__init__(brand, color)


b = BenZi('BenZi', 'red')
b.hello()
