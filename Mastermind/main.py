# Python Background
import random
sNum = str(random.randrange(1000,9999))
#print(sNum)
# isnum()
count = 1
usserG = input("Guess my four digit number: ")

#GuessStr = usserG
#print([*GuessStr])

while int(usserG) != int(sNum):
  holding = ['X', 'X', 'X', 'X']
  correct = 0
  for i in range(len(usserG)):
    if(usserG[i]==sNum[i]):
      holding[i] = usserG[i]
      correct += 1
  count+=1
  print("Not quite", "\nHere are the numbers you got right\n", holding)
  usserG = input("\nTry again: ")
  
# Step 5
if int(usserG) == int(sNum):
	print("Congrats you guessed the number in ", count, " tries!")


