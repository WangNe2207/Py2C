import random
def CreateTestTXT():

    #Open a file named numbersmake.txt.
    outfile = open('Input.txt', 'w')

    #Produce the numbers
    for count in range(3072):
        #Get a random number.
        num = random.randint(0, 255)
        outfile.write(str(num))
        outfile.write(' ')

    #Write 12 random intergers in the range of 1-100 on one line
    #to the file.


    #Close the file.
    outfile.close()
    print('Data written to Input.txt')

#Call the main function
CreateTestTXT()
