import bmp
class ImageProcessor:

    def __init__(self, filename): #Method 1, gets the height and width from the BMP function
        self.pixelgrid = bmp.ReadBMP(filename)
        self.height = len(self.pixelgrid)
        self.width = len(self.pixelgrid[0])


    def save(self, newName): #Method 2, saves desired changes to the newName file, by calling WriteBMP
        bmp.WriteBMP(self.pixelgrid, newName)


    def invert(self): #Method 3, inverts the colors in the image by subtracting the corresponding r,g, and b values from the maximum value, 255.
        for row in range(int(self.height)):
            for column in range(int(self.width)):
                for channel in range(3):
                    self.pixelgrid[row][column][0] = 255 - self.pixelgrid[row][column][0]
                    self.pixelgrid[row][column][1] = 255 - self.pixelgrid[row][column][1]
                    self.pixelgrid[row][column][2] = 255 - self.pixelgrid[row][column][2]


    def displayChannel(self, channel): #Method 4, changes r,g, and b respectively to analyze desired outcome
        for row in range(int(self.height)):
            for column in range(int(self.width)):
                if channel == 'r':
                    self.pixelgrid[row][column][1] = 0
                    self.pixelgrid[row][column][2] = 0
                elif channel == 'g':
                    self.pixelgrid[row][column][0] = 0
                    self.pixelgrid[row][column][2] = 0
                elif channel == 'b':
                    self.pixelgrid[row][column][0] = 0
                    self.pixelgrid[row][column][1] = 0


    def flip(self,axis): #Method 5, flips the image depeding on which flip is selected via the reverse function
        if axis == 'h':
            for row in range(self.height):
                self.pixelgrid[row].reverse()
        elif axis == 'v':
            self.pixelgrid.reverse()


    def grayscale(self): #Method 6, converts the image to grayscale by adding the r, g, and b values for each pixel and dividing by 3, rounding down
        for row in range(self.height):
            for column in range(self.width):
                gray = ((self.pixelgrid[row][column][0] + self.pixelgrid[row][column][1] + self.pixelgrid[row][column][2])//3)
                for channel in range(3):
                    self.pixelgrid[row][column][channel] = gray


    def brightness(self, operation): #Method 7, increases and decreases the brightness upon the user's input and loops until q is inputed
        if operation == '+':
            c = 25
        elif operation == '-':
            c = -25
        for row in range(int(self.height)):
            for column in range(int(self.width)):
                for channel in range(3):
                    newVal = int(self.pixelgrid[row][column][channel]+c)
                    if newVal > 255:
                        newVal = 255
                    elif newVal < 0:
                        newVal = 0
                    self.pixelgrid[row][column][channel] = newVal


    def contrast (self, operation): #Method 8, applies contrast from user's desired input and loops until the user presses q
        if operation == '+':
            c = 45
        elif operation == '-':
            c = -45
        factor = (259*(c+255))/(255*(259-c))
        for row in range(int(self.height)):
            for column in range(int(self.width)):
                for channel in range(3):
                    newVal = int(factor*(self.pixelgrid[row][column][channel]-128)+128)
                    if newVal > 255:
                        newVal = 255
                    elif newVal < 0:
                        newVal = 0
                    self.pixelgrid[row][column][channel] = newVal


def menu(): #Displays and verifies menu input
    menu = '''    =======================
    Select an Option:
    =======================
    a) Invert Colors
    b) Flip Image
    c) Display Color Channel
    d) Convert to Grayscale
    e) Adjust Brightness
    f) Adjust Contrast
    ----------------------
    s) Save Image
    o) Open new Image
    q) Quit
    =======================
    (a/b/c/d/e/f/s/o/q): '''
    uChoice = input(menu)
    while uChoice not in ('a','b','c','d','e','f','s','o','q'):
        print('Invalid input, please enter a,b,c,d,e,f,s,o, or q.')
        uChoice = input(menu)
    return uChoice


def flipInput(): #Verifies flip menu input
    axis = input('Would you like to flip horizontally (h), or vertically, (v)?')
    while axis not in ('h','v'):
        print("Invalid input, please enter h or v.")
        axis = input('Would you like to flip horizontally (h), or vertically, (v)?')
    return axis


def rgbInput(): #Verifies rgb menu input
    rgb =  input('Would you like to display red (r), green (g), or blue (b)?')
    while rgb not in ('r','g','b'):
        print('Invalid input please enter r, g, or b')
        rgb =  input('Would you like to display red (r), green (g), or blue (b)?')
    return rgb


def operationInput(): #Verifies operation menu input for brightness and contrast
    operation = input('Would you like to increase (+) or decrease (-) the brightness or contrast? Press (q) to quit.')
    while operation not in ('+', '-', 'q'):
        print('Invalid input please enter +, -, or q')
        operation = input('Would you like to increase (+) or decrease (-) the brightness or contrast? Press (q) to quit.')
    return operation


def menuInput(): #Central function that will call other functions depending on the user input
    uChoice = menu() #Prompts the menu first, user must enter a valid input
    while uChoice != 'o': #if the first input is not 'o', the function will loop until o is selected and a file is opened
        print("Please select 'o'.")
        uChoice = menu()
    while uChoice == 'o':
        filename = input("""Enter the .bmp file named you'd like to modify.""")
        textfile = open(filename, 'r+')
        my_processor = ImageProcessor(filename)
        uChoice = menu()
        while uChoice == 'a':
            my_processor.invert()
            uChoice = menu()
        while uChoice == 'b':
            axis = flipInput()
            my_processor.flip(axis)
            uChoice = menu()
        while uChoice == 'c':
            channel = rgbInput()
            my_processor.displayChannel(channel)
            uChoice = menu()
        while uChoice == 'd':
            my_processor.grayscale()
            uChoice = menu()
        while uChoice == 'e':
            operation = operationInput()
            while operation != 'q': #if the operation is q, it will prompt the menu to open again, same deal for uChoice == 'f'
                my_processor.brightness(operation)
                operation = operationInput()
            uChoice = menu()
        while uChoice == 'f':
            operation = operationInput()
            while operation != 'q':
                my_processor.contrast(operation)
                operation = operationInput()
            uChoice = menu()
        while uChoice == 's':
            newName = input('What would you like to name your new, modified .bmp file?')
            my_processor.save(newName)
            uChoice = menu()
        if uChoice == 'q':
            textfile.close()
            print('Quitting system', exit)


menuInput()


