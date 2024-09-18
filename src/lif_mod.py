from lava.proc.lif.process import LIF
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import AbstractPyDenseModelFloat
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType


import numpy as np


class LIFEncoder(LIF):
    pass

@implements(proc=LIFEncoder, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLifEncoderModelMixed(AbstractPyLifModelFloat):

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: float = LavaPyType(float, float)
    
    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return  self.v > self.vth
         
    
@implements(proc=LIFEncoder, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLifEncoderModelFloat(AbstractPyLifModelFloat):
    
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.vth