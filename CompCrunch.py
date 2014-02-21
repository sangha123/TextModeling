import numpy
import scipy
import re
from crunchbase import CrunchBase
from pymongo import MongoClient
from BeautifulSoup import BeautifulSoup
import urllib2
import pandas
import pickle
import unicodedata

if __name__ == "__main__":

    mycrunchbasekey = "c3awcmguzjhwnmqny8mzwtdn"

    cb = CrunchBase(mycrunchbasekey)
    client = MongoClient()
    db = client.crunch

    cc = pickle.load(open('compname.pkl', 'rb'))


    
    for comp in cc:
        compname=unicodedata.normalize('NFKD', 
            comp).encode('ascii', 'ignore')

        # print cmpnm
        if cb.getCompanyData(compname):
            m = cb.getCompanyData(compname)

            compdata = db.compdata

            compdata_id = compdata.insert(m)
