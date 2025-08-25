from lammps import lammps
import threading 
from mpi4py import MPI 
import queue 


class MpiApiEngine() : 
    """ 
    """
    
    def __init__(self, engine_comm: MPI.Comm, engine_id: int) -> None : 
        self.engine_comm = engine_comm 
        self.rank = engine_comm.Get_rank() #Ranks of the current process in the engine communicator 
        self.engine_id = engine_id #Identifier
        self.lmp = None #Placeholder for Lammps instance 
        self._is_alive = False 
        self._is_busy = False 
        self.message_reader_thread = None 
        self.message_queue = queue.Queue() #Queue to hold messages from the session 

    def start(self) -> None : 
        """Start Lammps"""
        self.start_engine() 
        self.start_reader_thread() 
        self.run_engine_loop() 

    def start_engine(self) -> None : 
        if self.engine_comm is None : 
            raise RuntimeError("Missing engine_comm for LAMMPS")

        self.lmp = lammps(comm=self.engine_comm, cmdargs=['-screen', 'none', '-log', 'lammps.log.'+str(self.engine_id)])
        self._is_alive = True

    def start_reader_thread(self) -> None : 
        """Start the message reader thread on the master rank"""
        if self.rank == 0 and self.message_reader_thread is None : 
            self.message_reader_thread = threading.Thread(target=self._read_messages, daemon=True)
            self.message_reader_thread.start()

    #RUN ON RANK 0
    def _read_messages(self):
        """Read messages from the LAMMPS session. it is run in a separate thread on the master rank.
        The message is a dictionnary of the form : {'type': str, 'value', str}"""
        while self._is_alive:
            #Check if a message is available from the session
            if MPI.COMM_WORLD.Iprobe(source=MPI.ANY_SOURCE) : 
                msg = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE)
                self.message_queue.put(msg)  # Add the message to the queue for processing

            else:
                # If no message is available, sleep for a short duration to avoid busy waiting
                threading.Event().wait(0.01)

    #RUN ON ALL RANKS
    def run_engine_loop(self):
        """All ranks run this while lammps is alive, when a message is broadcaster from the master rank, execute the command."""
        while self._is_alive:

            if self.rank == 0:
                try:
                    msg = self.message_queue.get(timeout=0.01)  # wait for a message
                except queue.Empty:
                    continue
            else:
                msg = None

            # broadcast to all ranks in engine_comm
            msg = self.engine_comm.bcast(msg, root=0)
            if msg is None:
                continue
            
            self._handle_message(msg)
            if self.rank == 0:
                self._send_status()
            #Check if engine was told to shutdown
            if not self._is_alive:
                break

    def _send_status(self) : 
        """Send the status of the engine to the session."""
        status_msg = {
        "type": "status",
        "value": {
            "alive": self._is_alive,
            "busy": self._is_busy,
            }
        }

        MPI.COMM_WORLD.send(status_msg, dest=0, tag=0)  # Rank 0 du world: session 


    def _handle_message(self, msg: dict) -> None:
        """Handle incoming messages from the session."""
        msg_type = msg.get("type")
        if msg_type == "close":
            self.engine_comm.barrier()  # synchronise every rank to close properly 
            self.close()
        elif msg_type == "command":
            cmd = msg.get("value")
            if cmd:
                self.engine_comm.barrier()  # Ensure all ranks are synchronized before sending the command
                try : 
                    self.command(cmd)
                except Exception as e:
                    print(f"[Engine Rank {self.rank}] Error executing command '{cmd}': {e}")
                self.engine_comm.barrier()  # Ensure all ranks are synchronized after sending the command

        elif msg_type == "partn" : 
            central_atom = msg.get("value")
            if central_atom is not None:
                try:
                    self.engine_comm.barrier()  # Ensure all ranks are synchronized before sending the command
                    self.partn(central_atom)
                    self.engine_comm.barrier()  # Ensure all ranks are synchronized after sending the command
                except Exception as e:
                    print(f"[Engine Rank {self.rank}] Error executing partn for atom {central_atom}: {e}")
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
        

    def command(self, cmd: str) -> None:
        """Send a command to the LAMMPS"""
        if self.lmp is None:
            raise RuntimeError("LAMMPS instance is not initialized. Please start the engine first.")
        if not self._is_alive:
            raise RuntimeError("LAMMPS process is not running.")
        if self._is_busy:
            raise RuntimeError("LAMMPS process is already busy.")

        self._is_busy = True
        try:
            self.lmp.command(cmd)
        except Exception as e:
            raise RuntimeError(f"Error executing command '{cmd}': {e}")
        finally:
            self._is_busy = False


    def close(self) -> None:    
        """Close the LAMMPS engine."""
        print(f"[Engine Rank {self.rank}] Closing LAMMPS engine.")
        if self.lmp is not None:
            self.lmp.close()
            self.lmp = None
        self._is_alive = False
        if self.message_reader_thread is not None:
            self.message_reader_thread.join(timeout=1)
            self.message_reader_thread = None

    def is_alive(self) -> bool:
        """Check if the LAMMPS engine is alive."""
        return self._is_alive       
    
    def is_busy(self) -> bool:
        """Check if the LAMMPS engine is busy."""
        return self._is_busy  