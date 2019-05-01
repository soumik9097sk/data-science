import random
nim = input('Enter your name : ')
name = list(nim.split(' '))
dayOB = str(input('Enter your date of birth(only the day) : '))
sentence = ['does charity work @ church in ', 
			'can be little extreme sometimes @ ',
			'is a very very generous person # ',
			'is the strongest person you see @ ' ,
			'is like thanos , everyone fears him * ',
			'is the most # cool guy but only on ',
			'can put you in the # worst nightmare in']
rand_sen = random.choice(sentence)
temp = rand_sen.split(' ')
firstN = []

for i in range(len(temp)-1):
    firstN.append(temp[i][:1])
nameFirst = []
for i in range(len(name)):
	nameFirst.append(name[i][:1])
firstN = ''.join(firstN)
nameFirst = ''.join(nameFirst)
password = nameFirst.upper()+firstN+dayOB

print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
print('Remember this line : \n'+nim,rand_sen,dayOB)
print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
print('The first character of each word including the symbols and the numbers constitute your password')
print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
print('your password : ',password)