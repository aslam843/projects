import mymodule as mx
from mymodule import greetings
from mymodule import person1
import platform
import datetime
import math
import json
import re

# Import the module named mymodule as mx (alias), and access the person1 dictionary:
mx.greetings("Shaikh")
print(mx.person1["age"])


# When importing using the from keyword, do not use the module name when referring to elements in the module. Example: person1["age"], not mymodule.person1["age"]
greetings("Aslam")
a = person1["name"] 
print(a)

# platform module import
x = platform.system()
print(x)

# Create a date object:
x1 = datetime.datetime(2020, 5, 17)
print(x1)

# Import the datetime module and display the current date:
y = datetime.datetime.now()
print(y)
print(y.year)
# strftime a method for formatting date objects into readable strings
print(y.strftime("%Y/%m/%d"))

# The math.sqrt() method for example, returns the square root of a number:
m = math.sqrt(64)
print(m)

x = math.ceil(1.4)
y = math.floor(1.4)
print(x) # returns 2
print(y) # returns 1

x = math.pi
print(x)


# Convert from JSON to Python:
x = '{"name":"John", "age":30, "city":"New York"}'
y = json.loads(x)
print(y["age"])

# Convert from Python to JSON: Use the indent parameter to define the numbers of indents:
x = {"name":"John", "age":30, "city":"New York"}
y = json.dumps(x, indent=4)
print(y)

# Search the string to see if it starts with "The" and ends with "Spain":
txt = "The rain in Spain"
x = re.search("^The.*Spain$", txt)
if x:
    print("Yes! We have match")
else:
    print("No match")

# The findall() function returns a list containing all matches.
x = re.findall("ai",txt)
print(x)

# Return an empty list if no match was found:
x = re.findall("Portugal", txt)
print(x)

# Split at each white-space character:
x = re.split("\s", txt)
print(x)

# Split the string only at the first occurrence:
x = re.split("\s", txt, 1)
print(x)

# Replace every white-space character with the number @:
x = re.sub("\s", "@", txt)
print(x)