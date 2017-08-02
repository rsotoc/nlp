# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:26:29 2016

@author: rsotoc
"""

import os
from bs4 import BeautifulSoup

def existe(filename) :
    try:
        os.stat(filename)
        ret = True
    except:
        ret = False
        
    return ret
    
               
def main (origen) :
    listing = os.listdir(origen)
    
    for l in listing :
        abs_origen = os.path.join(origen, l)
        file = open(abs_origen, encoding='iso-8859-1')
        raw = file.read()
        file.close()
        soup = BeautifulSoup(raw, "lxml")
        text = soup.get_text()
        print(text)
                                
if (__name__ == '__main__' ) :
    main("krahe")
    