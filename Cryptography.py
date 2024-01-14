import sys
import math
def removespace(string): #function that removes spaces from inputted text
    return string.replace(' ',"")


def adjusted_key(lowertext,lowerkey): #function that produces an adjusted key to the length of the given text
    textLength = len(removespace(lowertext))
    usedKey = removespace(lowerkey)
    keyLength = len(usedKey)
    repitition = math.ceil(textLength/keyLength)
    bigKey = usedKey * repitition
    endKey = bigKey[:textLength]
    return endKey #End of Part 1 Lab 1


def encrypt_vinegere(): #function asks user for desired key and text and then gives encrypted texts
    fileName = input('What is the name of the .txt file including the original text to be encrypted?') #File containing the message to be coded
    textfile = open(fileName,'r')
    text = str(textfile.readline())
    textfile.close()
    lowertext = text.lower()
    key = input('What key would you like to use to encrypt?')
    lowerkey = key.lower()
    endKey = adjusted_key(lowertext,lowerkey)
    listKey = list(endKey) #list for the adjusted key
    listText = list(text) # list for the text that was input
    encryptfilename = input('What would you like to name the .txt file containing the encrypted message?') #File that will contain encrypted code
    output_file = open(encryptfilename, 'w')
    j = 0 # variable that goes through the listKey and checks the position
    for i in range(len(listText)): #for loop used to encrypt the given text and key
        if 97 <= (ord(listText[i])) <= 122:
            codedtext = ((ord(listKey[j])-97) + (ord(listText[i])-97))%26 #Applies lowercase shift
            j += 1
            output_file.write(chr(codedtext+97))#Adds encrypted letters to list
        elif 65 <= (ord(listText[i])) <= 90:
            uppercodedtext = (((ord(listKey[j])-65)) + (ord(listText[i])-65))%26 #Applies uppercase shift
            j += 1
            output_file.write(chr(uppercodedtext+65))
        else:
            codednontext = ord(listText[i]) #Applies no shift for non letter characters
            output_file.write(chr(codednontext)) #End of Lab 1
    output_file.close()


def decrypt_vinegere():
    fileName = input('What is the name of the .txt file including the encrypted text?')
    textfile = open(fileName, 'r')
    text = str(textfile.readline())
    textfile.close()
    lowertext = text.lower()
    key = input('What is the decryption key used?')
    lowerkey = key.lower()
    endKey = adjusted_key(lowertext, lowerkey)
    listKey = list(endKey) #list for the adjusted key
    listText = list(text) # list for the text that was input
    newFile = input('What would you like to name the .txt file including the decypted text?')
    textfile2 = open(newFile, 'w')
    j = 0 # variable that goes through the listKey and checks the position
    for i in range(len(listText)): #for loop used to decrypt the given text and key
        if 97 <= (ord(listText[i])) <= 122:
            codedtext = ((ord(listText[i])-97)- (ord(listKey[j])-97))%26 #Applies lowercase shift
            j += 1
            textfile2.write(chr(codedtext+97)) #Adds decrypted letters to list
        elif 65 <= (ord(listText[i])) <= 90:
            uppercodedtext = ((ord(listText[i])-65) - (ord(listKey[j])-65))%26 #Applies uppercase shift
            j += 1
            textfile2.write(chr(uppercodedtext+65))
        else:
            codednontext = ord(listText[i]) #Applies no shift for non letter characters
            textfile2.write(chr(codednontext))#End of Part 1 Lab 2
    textfile2.close()


def menu(): #Asks for and validates the user's input
    menu = """     Select an Option:
    =====================
    (1) Encrypt Message
    (2) Decrypt Message
    (9) Exit Program
    =====================
     Select (1/2/9):""" #menu user can choose from
    uChoice = input(menu) #collects users input
    while uChoice not in ('1','2','9'): #if user enters an invald input, they're prompted to add a valid one
        print('Invalid input, please enter 1, 2, or 9')
        uChoice = input(menu)
    return uChoice


def main():
    uChoice = menu()
    while uChoice == '1': #if user enters 1, encrypt_vinegere is called
        encrypt_vinegere()
        uChoice = menu()
    while uChoice == '2': #if user enters 2, decrypt_vinegere is called.
        decrypt_vinegere()
        uChoice = menu()
    if uChoice == '9': #if the user enters 9 the program quits
            print('Exiting system', exit)
main()

