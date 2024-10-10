import bittensor as bt

metagraph = bt.subtensor("finney").metagraph(netuid=28)

print(metagraph.R)
print(metagraph.coldkeys)
print(metagraph.hotkeys)
