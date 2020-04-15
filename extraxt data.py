from PIL import Image
import binascii
import os


def is_prime(n):
    """ Advanced Prime Number Fn """
    if n == 2 or n == 3: return True
    if n < 2 or n%2 == 0: return False
    if n < 9: return True
    if n%3 == 0: return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n%f == 0: return False
        if n%(f+2) == 0: return False
        f +=6
    return True

def lsb(n):
    """ Return 2 last LSB as String """
    n = n%4
    if(n == 0):
        return "00"
    if(n == 1):
        return "01"
    if(n == 2):
        return "10"
    if(n == 3):
        return "11"

def StrHex2chr(hex):
    if len(hex)%2!=0:
        print('length of Str of hex should be even')
        hex=hex+'0'
        #return ''
    list_chr=[]
    for i in range(0,len(hex),2):
        hex2int=int(hex[i:i+2],16)
        if 32<=hex2int<=126:
            list_chr.append(chr(hex2int))
        else:
            list_chr.append('0x'+str(hex[i:i+2]))
    return ''.join(list_chr)


def main():
    os.chdir("C://Users//samaneh//Desktop//pixel indicator(30 point)")
    img = Image.open("cat.png")
    pxs = img.load() # 2D Array pxs[x,y] ; pxs[0,0] = top left
    w,h = img.size


    print("Load Image with size: "+str(w)+"*"+str(h))

    toct = []
    pxsl=[]

    #*********** calculate message size *****************

    for i in range(3):  # extract first 9 pxl to extract message length from 8 bytes 

        toct.append(format(pxs[i,0][0],'b'))    # format = on recup en binaire
        toct.append(format(pxs[i,0][1],'b'))
        pxsl.append((pxs[i,0][0]))
        pxsl.append((pxs[i,0][1]))
        pxsl.append((pxs[i,0][2]))
        if(i < 2):                              # On ne met pas le 9 octet
            toct.append(format(pxs[i,0][2],'b'))
    print('first 8 pixels =',pxsl[:-1])

    for i in range(len(toct)):                  #adding the missing "0" prefixes to complete 8 bits
        toct[i] = '0'*(8-len(toct[i]))+toct[i]

    print("first 8 bytes = ",toct)
    N = int(''.join(toct),2)                 #int of message size 
    print("Message size : "+str(N))

    #************** Determine Indicator Channel ******************

    if(N%2 == 0):
        IC = 0
        print("'N' is even ==> IC = 0 (R)")
    elif(is_prime(N)):   
        IC = 2
        print("'N' is prime ==> IC = 2 (B)")
    else:                
        IC = 1 
        print("'N' is neither even nor prime ==> IC =1 (G)")


    #************* find out Data channels by parity bits ******************
    IC=1
    """
    calculate the "binary parity":
    count of '1' bits even ==> 0
    count of '1' bits odd  ==> 1
    """

    parity = format(N,'b').count("1") # Count of '1' bits 

    if parity%2==0:
        print('\nparity is even')
        parity=0
    else:
        print('\nparity is odd')
        parity=1    


    c = ["R","G","B"]

    if(parity == 1): # Odd 

        if(IC == 0): 
            c1 = 1
            c2 = 2
        elif(IC == 2): 
            c1 = 0
            c2 = 1
        else: 
            c1 = 0
            c2 = 2
    else:           # Even
        if(IC == 0): 
            c1 = 2 
            c2 = 1 
        elif(IC == 2): 
            c1 = 1
            c2 = 0
        else: 
            c1 = 2
            c2 = 0

    print("-----------------")
    print("Indicator Chanel: "+c[IC])
    print("Data Channel 1: "+c[c1])
    print("Data Channel 2: "+c[c2])

    #************ Extracting Hidden Data ****************

    RB = N  # RB: Reminded Bits to extract
    pxsl = list(img.getdata())[w:] # image's pixels without first row

    binSecret = "" 
    i = 0 # current pixel 

    while(RB > 0):     # As long as we extract all bits

        indicLSB = bin(pxsl[i][IC])[-2:]  # Geting 2 LSB values of IC pixels
    
        if(indicLSB =='01'):        # Channel 2 is used to extract
            binSecret += lsb(pxsl[i][c2])
            RB -= 2
        elif(indicLSB =='10'):      # Channel 1 is used to extract
            binSecret += lsb(pxsl[i][c1])
            RB -= 2
        elif(indicLSB == '11'):     # Channels 1 and 2 are used
            binSecret += lsb(pxsl[i][c1])
            binSecret += lsb(pxsl[i][c2])
            RB -= 4
        i += 1

    a = hex(int(binSecret,2))[2:]   #Convert extract bits to hex
    print('\nHidden message: \n')
    print(StrHex2chr(a))

if __name__=='__main__':
    main()
