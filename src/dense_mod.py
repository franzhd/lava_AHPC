from lava.proc.dense.process import Dense
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import AbstractPyDenseModelFloat, PyDenseModelFloat
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

import numpy as np
import typing as ty

class DenseMod(Dense):
    pass

@implements(proc=DenseMod, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyDenseModModelFloat(AbstractPyDenseModelFloat):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons)in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def run_spk(self):
        # The a_out sent on a each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        if self.time_step <2:
            self.a_out.send(self.a_buff)
        else:
            
            if self.num_message_bits.item() > 0:
                s_in = self.s_in.recv()
                self.a_buff = self.weights.dot(s_in)
            else:
                s_in = self.s_in.recv().astype(bool)
                self.a_buff = self.weights[:, s_in].sum(axis=1)

            self.a_out.send(self.a_buff)

class DenseEncoder(Dense):
    pass

@implements(proc=DenseEncoder, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt", "floating_pt")
class PyDenseEncoderModelFloat(AbstractPyDenseModelFloat):
    pass

