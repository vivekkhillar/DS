from typing import TypedDict

# This will only use to define the structure of a person in a chat application or dictionary

class Person (TypedDict):
    
    First_Name: str
    Last_Name: str
    Age: int


new_person: Person = { 'First_Name': 'John', 'Last_Name': 'Khillar', 'Age' : 30}

print(new_person)