__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from inspect import isclass
from pybrain.tools.customxml import NetworkWriter as PyBrainNetworkWriter
from pybrain.utilities import canonicClassString
from numpy import ndarray, savetxt
from io import BytesIO

class NetworkWriter(PyBrainNetworkWriter):

    @staticmethod
    def writeToFile(net, filename):
        """ write the network as a new xml file """
        w = NetworkWriter(filename, newfile = True)
        netroot = w.newRootNode('Network')
        w.writeNetwork(net, netroot)
        w.save()

    def writeArgs(self, node, argdict):
        """ write a dictionnary of arguments """
        for name, val in argdict.items():
            if val != None:
                tmp = self.newChild(node, name)
                if isclass(val):
                    s = canonicClassString(val)
                elif isinstance(val, ndarray):
                    tmp.setAttribute('shape', 'x'.join([str(x) for x in val.shape]))
                    tmp.setAttribute('dtype', str(val.dtype))
                    output = BytesIO()
                    savetxt(output, val.view(float))
                    self.addTextNode(tmp, output.getvalue())
                    s = 'ndarray'
                else:
                    s = getattr(val, 'name', repr(val))
                tmp.setAttribute('val', s)