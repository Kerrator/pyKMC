from pykmc.system import System
system = System('input.in')
system.kmc()
system.catalog.to_pickle('catalog.pickle')
