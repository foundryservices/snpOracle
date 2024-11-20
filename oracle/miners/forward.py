

from oracle.protocol import Challenge


def forward(synapse: Challenge) -> Challenge:
    """
    How miners should process incoming synapses and respond to them.
    """
    
    synapse.prediction = [6000, 6001, 6002, 6003, 6004, 6005]

    return synapse
