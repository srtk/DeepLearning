#!/usr/bin/env python

#http://xnoiz.blogspot.jp/2010/01/python.html
#print getVarNames(b,locals())
def getVarNames(obj, preLocals):
    return [k for k, v in preLocals.items() if id(obj) == id(v)]
#    results = []
#    for k, v in preLocals.items():
#         if id(obj) == id(v):
#             results.append(k)
#    return results

#print getVarNamesG(a)
def getVarNamesG(obj):
    return [k for k, v in globals().items() if id(obj) == id(v)]


# In MacOSX, this method returns "maximum resident set size" in MBs
# In other systems, the unit may vary depends on what your system's "getrusage" returns
# see "man getrusage" for exact information
def maxMemoryUsed():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.*1024.)


def timestamp():
    import datetime
    now = datetime.datetime.today()
    return "%s/%s/%s %02d:%02d:%02d.%s" % (now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)

if __name__ == '__main__':
    a = [0,1,2]
    b = [2,3,4,5]
    pair = (a,b)
    if not (len(pair[0]) == len(pair[1])):
        namea = getVarNames(pair[0], locals())
        nameb = getVarNames(pair[1], locals())
        print("WARNING: the lengths of %s & %s are different" % (namea,nameb))
