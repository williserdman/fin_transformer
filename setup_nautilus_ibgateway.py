from nautilus_trader.adapters.interactive_brokers.config import (
    DockerizedIBGatewayConfig,
)
from nautilus_trader.adapters.interactive_brokers.gateway import DockerizedIBGateway
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from current working directory

gateway_config = DockerizedIBGatewayConfig(
    username=os.environ["TWS_USERNAME"],
    password=os.environ["TWS_PASSWORD"],
    trading_mode="paper",
    read_only_api=False,
    # timeout=300
)

# This may take a short while to start up, especially the first time
gateway = DockerizedIBGateway(config=gateway_config)
gateway.start()

# Confirm you are logged in
print(gateway.is_logged_in(gateway.container))

# Inspect the logs
print(gateway.container.logs())
