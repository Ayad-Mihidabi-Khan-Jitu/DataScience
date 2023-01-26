print("Enter a letter")
dog=input()
if(dog=='A' or dog=='E' or dog=='I' or dog=='O'or dog=='U'or dog=='a'or dog=='e'or dog=='i'or dog=='o'or dog=='u' ):  
    print("dog=vowel")
elif((dog>='A' and  dog<='z')and(dog!='A' or dog!='E' or dog!='I' or dog!='O'or dog!='U'or dog!='a'or dog!='e'or dog!='i'or dog!='o'or dog!='u' )):
    print("dog=consonant") 
else:
    print("this is not a letter")    