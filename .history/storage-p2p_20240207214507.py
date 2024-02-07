#!/usr/bin/env python
import concurrent.futures
import argparse
import configparser
import logging
import random
from dataclasses import dataclass
from random import expovariate
from typing import Optional, List
from abc import abstractmethod
import time

# the humanfriendly library (https://humanfriendly.readthedocs.io/en/latest/) lets us pass parameters in human-readable
# format (e.g., "500 KiB" or "5 days"). You can safely remove this if you don't want to install it on your system, but
# then you'll need to handle sizes in bytes and time spans in seconds--or write your own alternative.
# It should be trivial to install (e.g., apt install python3-humanfriendly or conda/pip install humanfriendly).
from humanfriendly import format_timespan, parse_size, parse_timespan

from discrete_event_sim import Simulation, Event
import matplotlib.pyplot as plt


def exp_rv(mean):
    """Return an exponential random variable with the given mean."""
    return expovariate(1 / mean)


class DataLost(Exception):
    """Not enough redundancy in the system, data is lost. We raise this exception to stop the simulation."""
    pass


class Backup(Simulation):
    """Backup simulation.
    """

    # type annotations for `Node` are strings here to allow a forward declaration:
    # https://stackoverflow.com/questions/36193540/self-reference-or-forward-reference-of-type-annotations-in-python
    def __init__(self, nodes: List['Node']):
        super().__init__()  # call the __init__ method of parent class
        self.nodes = nodes
        self.backed_up_blocks_count = [0] * len(nodes) 
        self.restored_blocks_count = [0] * len(nodes)
        # we add to the event queue the first event of each node going online and of failing
        for node in nodes:
            self.schedule(node.arrival_time, Online(node))
            self.schedule(node.arrival_time + exp_rv(node.average_lifetime), Fail(node))
             
    def schedule_transfer(self, uploader: 'Node', downloader: 'Node', block_id: int, restore: bool):
        """Helper function called by `Node.schedule_next_upload` and `Node.schedule_next_download`.

        If `restore` is true, we are restoring a block owned by the downloader, otherwise, we are saving one owned by
        the uploader.
        """

        block_size = downloader.block_size if restore else uploader.block_size

        assert uploader.current_upload is None
        assert downloader.current_download is None

        speed = min(uploader.upload_speed, downloader.download_speed)  # we take the slowest between the two
        delay = block_size / speed
        if restore:
            event = BlockRestoreComplete(uploader, downloader, block_id)
        else:
            event = BlockBackupComplete(uploader, downloader, block_id)
        
        self.schedule(delay, event)
        uploader.current_upload = downloader.current_download = event

        # self.log_info(f"scheduled {event.__class__.__name__} from {uploader} to {downloader}"
        #               f" in {format_timespan(delay)}")

    def log_info(self, msg):
        """Override method to get human-friendly logging for time."""

        logging.info(f'{format_timespan(self.t)}: {msg}')


@dataclass(eq=False)  # auto initialization from parameters below (won't consider two nodes with same state as equal)
class Node:
    """Class representing the configuration of a given node."""

    # using dataclass is (for our purposes) equivalent to having something like
    def __init__(self, name,id, n, k, data_size, storage_size, upload_speed, download_speed, average_uptime, average_downtime, average_lifetime, average_recover_time, arrival_time):
        self.name = name
        self.id = id
        self.n = n
        self.k = k
        self.data_size = data_size
        self.storage_size = storage_size
        self.upload_speed = upload_speed
        self.download_speed = download_speed
        self.average_uptime = average_uptime
        self.average_downtime = average_downtime
        self.average_lifetime = average_lifetime
        self.average_recover_time = average_recover_time
        self.arrival_time = arrival_time
        
        self.__post_init__()  # if the method exists

    name: str  # the node's name
    id : int   # unique identifier for this node
    n: int  # number of blocks in which the data is encoded
    k: int  # number of blocks sufficient to recover the whole node's data

    data_size: int  # amount of data to back up (in bytes)
    storage_size: int  # storage space devoted to storing remote data (in bytes)

    upload_speed: float  # node's upload speed, in bytes per second
    download_speed: float  # download speed

    average_uptime: float  # average time spent online
    average_downtime: float  # average time spent offline

    average_lifetime: float  # average time before a crash and data loss
    average_recover_time: float  # average time after a data loss

    arrival_time: float  # time at which the node will come online

    def __post_init__(self):
        """Compute other data dependent on config values and set up initial state."""

        # whether this node is online. All nodes start offline.
        self.online: bool = False

        # whether this node is currently under repairs. All nodes are ok at start.
        self.failed: bool = False

        # size of each block
        self.block_size: int = self.data_size // self.k if self.k > 0 else 0

        # amount of free space for others' data -- note we always leave enough space for our n blocks
        self.free_space: int = self.storage_size - self.block_size * self.n

        assert self.free_space >= 0, "Node without enough space to hold its own data"

        # local_blocks[block_id] is true if we locally have the local block
        # [x] * n is a list with n references to the object x
        self.local_blocks: list[bool] = [True] * self.n

        # backed_up_blocks[block_id] is the peer we're storing that block on, or None if it's not backed up yet;
        # we start with no blocks backed up
        self.backed_up_blocks: list[Optional[Node]] = [None] * self.n

        # (owner -> block_id) mapping for remote blocks stored
        self.remote_blocks_held: dict[Node, int] = {}

        # current uploads and downloads, stored as a reference to the relative TransferComplete event
        self.current_upload: Optional[TransferComplete] = None
        self.current_download: Optional[TransferComplete] = None
    
    
    def find_block_to_back_up(self):
        """Returns the block id of a block that needs backing up, or None if there are none."""

        # find a block that we have locally but not remotely
        # check `enumerate` and `zip`at https://docs.python.org/3/library/functions.html
        for block_id, (held_locally, peer) in enumerate(zip(self.local_blocks, self.backed_up_blocks)):
            if held_locally and peer is None:
                return block_id
        return None

    def schedule_next_upload(self, sim: Backup):
        """Schedule the next upload, if any."""

        assert self.online

        if self.current_upload is not None:
            return

        # first find if we have a backup that a remote node needs
        for peer, block_id in self.remote_blocks_held.items():
            # if the block is not present locally and the peer is online and not downloading anything currently, then
            # schedule the restore from self to peer of block_id
            if peer.online and peer.current_download is None and not peer.local_blocks[block_id]:
                sim.schedule_transfer(self,peer,block_id,True)
                sim.restored_blocks_count[peer.id]+= 1
                return  # we have found our upload, we stop

        # try to back up a block on a locally held remote node
        block_id = self.find_block_to_back_up()
        if block_id is None:
            return
        # sim.log_info(f"{self} is looking for somebody to back up block {block_id}")
        remote_owners = set(node for node in self.backed_up_blocks if node is not None)  # nodes having one block
        for peer in sim.nodes:
            # if the peer is not self, is online, is not among the remote owners, has enough space and is not
            # downloading anything currently, schedule the backup of block_id from self to peer
            if (peer is not self and peer.online and peer not in remote_owners and peer.current_download is None
                    and peer.free_space >= self.block_size):
                sim.schedule_transfer(self,peer,block_id,False)
                sim.backed_up_blocks_count[self.id]+=1
                return

    def schedule_next_download(self, sim: Backup):
        """Schedule the next download, if any."""

        assert self.online

        # sim.log_info(f"schedule_next_download on {self}")

        if self.current_download is not None:
            return

        # first find if we have a missing block to restore
        for block_id, (held_locally, peer) in enumerate(zip(self.local_blocks, self.backed_up_blocks)):
            if not held_locally and peer is not None and peer.online and peer.current_upload is None:
                sim.schedule_transfer(peer,self,block_id,True)
                sim.restored_blocks_count[self.id]+= 1

                return  # we are done in this case

        # try to back up a block for a remote node
        for peer in sim.nodes:
            if (peer is not self and peer.online and peer.current_upload is None and peer not in self.remote_blocks_held.keys()
                    and self.free_space >= peer.block_size):
                block_id = peer.find_block_to_back_up()
                if block_id is not None:
                    sim.schedule_transfer(peer,self,block_id,False)
                    sim.backed_up_blocks_count[peer.id]+=1

                    return

    def __hash__(self):
        """Function that allows us to have `Node`s as dictionary keys or set items.

        With this implementation, each node is only equal to itself.
        """
        return id(self)

    def __str__(self):
        """Function that will be called when converting this to a string (e.g., when logging or printing)."""

        return self.name


@dataclass
class NodeEvent(Event):
    """An event regarding a node. Carries the identifier, i.e., the node's index in `Backup.nodes_config`"""

    node: Node

    def process(self, sim: Simulation):
        """Must be implemented by subclasses."""
        raise NotImplementedError


class Online(NodeEvent):
    """A node goes online."""

    def process(self, sim: Backup):
        node = self.node
        if node.online or node.failed:
            return
        node.online = True
        # schedule next upload and download
        node.schedule_next_upload(sim)
        node.schedule_next_download(sim)
        
        # schedule the next offline event
        sim.schedule(exp_rv(node.average_lifetime),Offline(node))


class Recover(Online):
    """A node goes online after recovering from a failure."""

    def process(self, sim: Backup):
        node = self.node
        sim.log_info(f"{node} recovers")
        node.failed = False
        super().process(sim)
        sim.schedule(exp_rv(node.average_lifetime), Fail(node))


class Disconnection(NodeEvent):
    """Base class for both Offline and Fail, events that make a node disconnect."""

    def process(self, sim: Simulation):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def disconnect(self):
        node = self.node
        node.online = False
        # cancel current upload and download
        # retrieve the nodes we're uploading and downloading to and set their current downloads and uploads to None
        current_upload, current_download = node.current_upload, node.current_download
        if current_upload is not None:
            current_upload.canceled = True
            current_upload.downloader.current_download = None
            node.current_upload = None
        if current_download is not None:
            current_download.canceled = True
            current_download.uploader.current_upload = None
            node.current_download = None


class Offline(Disconnection):
    """A node goes offline."""

    def process(self, sim: Backup):
        node = self.node
        if node.failed or not node.online:
            return
        assert node.online
        self.disconnect()
        # schedule the next online event
        sim.schedule(exp_rv(self.node.average_downtime), Online(node))


class Fail(Disconnection):
    """A node fails and loses all local data."""

    def process(self, sim: Backup):
        sim.log_info(f"{self.node} fails")
        self.disconnect()
        node = self.node
        node.failed = True
        node.local_blocks = [False] * node.n  # lose all local data
        # lose all remote data
        for owner, block_id in node.remote_blocks_held.items():
            owner.backed_up_blocks[block_id] = None
            if owner.online and owner.current_upload is None:
                owner.schedule_next_upload(sim)  # this node may want to back up the missing block
        node.remote_blocks_held.clear()
        node.free_space = node.storage_size - node.block_size * node.n
        # schedule the next online and recover events
        recover_time = exp_rv(node.average_recover_time)
        sim.schedule(recover_time, Recover(node))


@dataclass
class TransferComplete(Event):
    """An upload is completed."""

    uploader: Node
    downloader: Node
    block_id: int
    canceled: bool = False

    def __post_init__(self):
        assert self.uploader is not self.downloader

    def process(self, sim: Backup):
        sim.log_info(f"{self.__class__.__name__} from {self.uploader} to {self.downloader}")
        if self.canceled:
            return  # this transfer was canceled, so ignore this event
        uploader, downloader = self.uploader, self.downloader
        assert uploader.online and downloader.online
        self.update_block_state()
        uploader.current_upload = downloader.current_download = None
        uploader.schedule_next_upload(sim)
        downloader.schedule_next_download(sim)
        #sim.count_backed_up_blocks
        for node in [uploader, downloader]:
            sim.log_info(f"{node}: {sum(node.local_blocks)} local blocks, "
                         f"{sum(peer is not None for peer in node.backed_up_blocks)} backed up blocks, "
                         f"{len(node.remote_blocks_held)} remote blocks held")

    def update_block_state(self):
        """Needs to be specified by the subclasses, `BackupComplete` and `DownloadComplete`."""
        raise NotImplementedError


class BlockBackupComplete(TransferComplete):

    def update_block_state(self):
        owner, peer = self.uploader, self.downloader
        peer.free_space -= owner.block_size
        assert peer.free_space >= 0
        owner.backed_up_blocks[self.block_id] = peer
        peer.remote_blocks_held[owner] = self.block_id
        
        


class BlockRestoreComplete(TransferComplete):
    def update_block_state(self):
        owner = self.downloader
        owner.local_blocks[self.block_id] = True
        if sum(owner.local_blocks) == owner.k:  # we have exactly k local blocks, we have all of them then
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="p2p.cfg", help="configuration file")
    parser.add_argument("--max-t", default="100 years")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout

    # functions to parse every parameter of peer configuration
    parsing_functions = [
        ('id', int), ('n', int), ('k', int),
        ('data_size', parse_size), ('storage_size', parse_size),
        ('upload_speed', parse_size), ('download_speed', parse_size),
        ('average_uptime', parse_timespan), ('average_downtime', parse_timespan),
        ('average_lifetime', parse_timespan), ('average_recover_time', parse_timespan),
        ('arrival_time', parse_timespan)
    ]
    count_backup=[[0, 0] for _ in range(11)]
    count_restore=[[0, 0] for _ in range(11)]
    for _ in range(0,10):
        vr=3
        r_values = [[0, 0] for _ in range(11)]
        counter_r = 0
        b_values = [[0, 0] for _ in range(11)]
        counter_b = 0
        r_vs_b = [[0, 0] for _ in range(11)]

        counter =0
        
        for r in range(10,120,10):
            config = configparser.ConfigParser()
            config.read(args.config)
            nodes = []  # we build the list of nodes to pass to the Backup class

            for node_class in config.sections():
                class_config = config[node_class]
                # list comprehension: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
                cfg = [parse(class_config[name]) for name, parse in parsing_functions]
                # the `callable(p1, p2, *args)` idiom is equivalent to `callable(p1, p2, args[0], args[1], ...)
                nodes.extend(Node(f"{node_class}-{i}", *cfg) for i in range(class_config.getint('number')))
            sim = Backup(nodes)
            for y in range(len(nodes)):
                sim.nodes[y].id = y
                if(y%vr==0):
                    sim.nodes[y].average_lifetime *= r
            sim.run(parse_timespan(args.max_t))
            sim.log_info(f"Simulation over")
         
            var_ratio_r = sum(sim.restored_blocks_count[i] for i in range(len(sim.restored_blocks_count)) if i % vr == 0)
            nonvar_ratio_r = sum(sim.restored_blocks_count[i] for i in range(len(sim.restored_blocks_count)) if i % vr != 0)
            sum_r=sum(sim.restored_blocks_count)
            r_values[counter_r][0] = var_ratio_r/sum_r
            r_values[counter_r][1] = nonvar_ratio_r/sum_r
            counter_r +=1
            var_ratio_b = sum(sim.backed_up_blocks_count[i] for i in range(len(sim.backed_up_blocks_count)) if i % vr == 0)
            nonvar_ratio_b = sum(sim.backed_up_blocks_count[i] for i in range(len(sim.backed_up_blocks_count)) if i % vr != 0)
            sum_b=sum(sim.backed_up_blocks_count)
            b_values[counter_b][0] = var_ratio_b/sum_b
            b_values[counter_b][1] = nonvar_ratio_b/sum_b
            counter_b +=1
            r_vs_b[counter]=sum(sim.restored_blocks_count)/ sum(sim.backed_up_blocks_count)
            counter+=1
        
            # Plot the ratio
        for i in range(0,11):
            count_restore[i][0] += r_values[i][0]
            count_restore[i][1] += r_values[i][1]
            count_backup[i][0] += b_values[i][0]
            count_backup[i][1] += b_values[i][1]
        print(f" count of restor:{sim.restored_blocks_count}")
        print(f" count of backup:{sim.backed_up_blocks_count}")
    for i in range(0,11):
        count_restore[i][0] /= 10
        count_restore[i][1] /= 10
        count_backup[i][0] /= 10
        count_backup[i][1] /= 10
    plt.plot(range(10, 120, 10), [r[0] for r in count_restore], linestyle='-', color='blue', label='Var Restored Blocks')
    #plt.plot(range(10, 120, 10), [r[1] for r in count_restore], linestyle='-', color='purple', label='NonVar Restored Blocks')    
    plt.plot(range(10, 120, 10), [b[0] for b in count_backup], linestyle='--', color='red', label='Var Backed Up Blocks')
    plt.plot(range(10, 120, 10), [b[1] for b in count_backup], linestyle='--', color='black', label='Non Var Backed Up Blocks')
    plt.plot(range(10, 120, 10), r_vs_b, linestyle=':', color='green', label='ratio')
    plt.xlabel('40% of nodes have a variant lifetime ')
    plt.ylabel('ratio of restored blocks that have variant life time with the rest of the nodes')
    plt.title(f"40% variant")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.xlim(0, 120)
    plt.xticks(range(0, 120, 10))
    plt.show()
if __name__ == '__main__':
    main()
