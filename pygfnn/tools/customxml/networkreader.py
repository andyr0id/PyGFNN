__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

# those imports are necessary for the eval() commands to find the right classes
import pygfnn #@UnusedImport
import pybrain #@UnusedImport
from scipy import array #@UnusedImport
from pybrain.tools.customxml import NetworkReader as PyBrainNetworkReader

class NetworkReader(PyBrainNetworkReader):

    @staticmethod
    def readFrom(filename, name = None, index = 0):
        """ append the network to an existing xml file

        :key name: if this parameter is specified, read the network with this name
        :key index: which network in the file shall be read (if there is more than one)
        """
        r = NetworkReader(filename, newfile = False)
        if name:
            netroot = r.findNamedNode('Network', name)
        else:
            netroot = r.findNode('Network', index)

        return r.readNetwork(netroot)

    def readNetwork(self, node):
        # TODO: why is this necessary?
        import pybrain.structure.networks.custom #@Reimport @UnusedImport
        nclass = eval(str(node.getAttribute('class')))
        argdict = self.readArgs(node)
        n = nclass(**argdict)
        n.name = node.getAttribute('name')

        for mnode in self.getChildrenOf(self.getChild(node, 'Modules')):
            m, inmodule, outmodule = self.readModule(mnode)
            if inmodule:
                n.addInputModule(m)
            elif outmodule:
                n.addOutputModule(m)
            else:
                n.addModule(m)

        mconns = self.getChild(node, 'MotherConnections')
        if mconns:
            for mcnode in self.getChildrenOf(mconns):
                m = self.readBuildable(mcnode)
                self.mothers[m.name] = m

        for cnode in self.getChildrenOf(self.getChild(node, 'Connections')):
            c, recurrent = self.readConnection(cnode)
            if recurrent:
                n.addRecurrentConnection(c)
            else:
                n.addConnection(c)

        n.sortModules()
        return n

    def readBuildable(self, node):
        mclass = node.getAttribute('class')
        argdict = self.readArgs(node)
        try:
            m = eval(mclass)(**argdict)
        except:
            print('Could not construct', mclass)
            print('with arguments:', argdict)
            return None
        m.name = node.getAttribute('name')
        self.readParams(node, m)
        return m

    def readArgs(self, node):
        res = {}
        for c in self.getChildrenOf(node):
            val = c.getAttribute('val')
            if val in self.modules:
                res[str(c.nodeName)] = self.modules[val]
            elif val in self.mothers:
                res[str(c.nodeName)] = self.mothers[val]
            elif val != '':
                res[str(c.nodeName)] = eval(val)
        return res

    def readParams(self, node, m):
        import string
        pnode = self.getChild(node, 'Parameters')
        if pnode:
            params = eval(string.strip(pnode.firstChild.data))
            m._setParameters(params)