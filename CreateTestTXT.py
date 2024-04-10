import random
def CreateTestTXT():

    #Open a file named numbersmake.txt.
    outfile = open('Test_flatten.txt', 'w')

    #71919 3072
    #Produce the numbers
    for count in range(13 * 6 * 6):
        #Get a random number.
        num = count / 10
        outfile.write(str(num))
        outfile.write(' ')

    #Write 12 random intergers in the range of 1-100 on one line
    #to the file.


    #Close the file.
    outfile.close()
    print('Data written to Input.txt')

#Call the main function
CreateTestTXT()
