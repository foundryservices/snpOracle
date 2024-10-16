import json

__version__ = "2.1.1"
version_split = __version__.split(".")
__spec_version__ = (1000 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))

SUBNET_LINKS = None
with open("subnet_links.json") as f:
    links_dict = json.load(f)
    SUBNET_LINKS = links_dict.get("subnet_repositories", None)
