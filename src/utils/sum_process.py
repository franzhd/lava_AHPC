import os
import numpy as np
import typing as ty

# Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

# Import parent classes for ProcessModels
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires, tag

class SumProcess(AbstractProcess):
    """Process to sum up the input spikes"""
    def __init__(self,shape, op, **kwargs):
        super().__init__()
        self.a_1_in = InPort(shape=shape)
        self.a_2_in = InPort(shape=shape)
        self.a_out = OutPort(shape=shape)
        self.op = Var(shape=shape, init=op)

@implements(proc=SumProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PySumProcessModel(PyLoihiProcessModel):
    a_1_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_2_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    op: int = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def run_spk(self):

        if self.time_step <3:
            self.out = np.zeros(self.a_1_in.shape)
            self.a_out.send(self.out)
        else:
            a_1 = self.a_1_in.recv()
            a_2 = self.a_2_in.recv()
            if self.op == 0:
                self.out = a_1 + a_2
            elif self.op == 1:
                self.out = a_1 - a_2

            self.a_out.send(self.out)