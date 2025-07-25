import sys
import numpy as np

print(sys.version)

fruits = ["apple ", "banana ", "cherry "]
x,y,z = fruits
print(x,y,z)
print(x+y+z)

arr = np.array([1, 2, 3, 4, 5])

print(arr)


x="something"

def getp():	
	#global x
	x = "fantastic"
	print("I am looking for " + x)

getp()
print("I am not looking for " + x)


thislist = list(("apple", "banana", "cherry")) # note the double round-brackets
print(thislist)

#list example
listr = ["apple","banana","cherry","cherry"]
listr[0] = "pineapple"
print(listr)
print(type(listr))

#tuple exmple
tpl = ("apple","banana","cherry","cherry")
#tpl[0] = "pineapple"
print(tpl)
print(type(tpl))

#set example
setr = {"apple","banana","cherry","cherry"}
#setr[0] = "pineapple"
print(setr)
print(type(setr))

#dictionary example
disct = {1:"apple",2:"banana",3:"banana"}
disct[1] = "pineapple"
print(disct)
print(type(disct))
