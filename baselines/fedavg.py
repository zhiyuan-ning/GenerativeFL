from plato.servers import fedavg
from plato.config import Config
from plato.datasources import base
from plato.trainers import basic
from plato.clients import simple



def main():
    """Starting point for a Plato federated learning training session."""
    server = fedavg.Server()
    server.run()


if __name__ == "__main__":
    main()